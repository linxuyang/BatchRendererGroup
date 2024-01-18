using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

public class MachineGun : MonoBehaviour
{
    public Mesh m_mesh;
    public Material m_material;
    public int bulletCount;
    public float speed = 10;
    private const int kGpuItemSize = (3 * 2 + 1) * 16; //  每个实例字节数 ( 2 * 4x3 matrices + 1 color per item )

    private BRG_Container m_brgContainer;
    private JobHandle m_updateJobFence;

    private struct BackgroundItem
    {
        public float x;
        public float y;
        public float z;
        public float cloudY;
        public float3x3 mat;
        public Vector4 color;
    };

    private NativeArray<BackgroundItem> m_backgroundItems;

    void Start()
    {
        m_brgContainer = new BRG_Container();
        m_brgContainer.Init(m_mesh, m_material, bulletCount, kGpuItemSize, false);

        m_backgroundItems = new NativeArray<BackgroundItem>(bulletCount, Allocator.Persistent, NativeArrayOptions.ClearMemory);

        InjectNewSlice();

        m_brgContainer.UploadGpuData(bulletCount);
    }

    [BurstCompile]
    private void InjectNewSlice()
    {
        var index = 0;
        for (int i = 0; i < bulletCount; i++)
        {
            float x = Random.Range(12, 30.0f);
            float z = Random.Range(-500, 0);
            float y = Random.Range(3.0f, 10.0f);
            BackgroundItem item = new BackgroundItem();
            item.x = x;
            item.y = y;
            item.z = z;
            item.cloudY = 100;
            item.color = new Vector4(Random.Range(0.0f, 1.0f), Random.Range(0.0f, 1.0f), Random.Range(0.0f, 1.0f), 1.0f);
            item.mat = float3x3.Scale(0.35f, 0.35f, 1);
            m_backgroundItems[index] = item;
            index++;
        }
    }

    [BurstCompile]
    private struct UpdatePositionsJob : IJobFor
    {
        [WriteOnly] [NativeDisableParallelForRestriction]
        public NativeArray<float4> _sysmemBuffer;

        [NativeDisableParallelForRestriction] public NativeArray<BackgroundItem> backgroundItems;

        public int _maxInstancePerWindow;
        public int _windowSizeInFloat4;
        public float _dt;
        public bool _change;
        public float _speed;

        public void Execute(int index)
        {
            var item = backgroundItems[index];
            var itemY = item.y;
            if (item.z < 0 || item.z > 100)
            {
                itemY = item.cloudY;
            }
            int windowId = 0; //System.Math.DivRem(slice * backgroundW + x, _maxInstancePerWindow, out i);
            int windowOffsetInFloat4 = windowId * _windowSizeInFloat4;
            float3x3 rot = item.mat;
            // compute the new current frame matrix
            _sysmemBuffer[(windowOffsetInFloat4 + index * 3 + 0)] = new float4(rot.c0.x, rot.c0.y, rot.c0.z, rot.c1.x);
            _sysmemBuffer[(windowOffsetInFloat4 + index * 3 + 1)] = new float4(rot.c1.y, rot.c1.z, rot.c2.x, rot.c2.y);
            _sysmemBuffer[(windowOffsetInFloat4 + index * 3 + 2)] = new float4(rot.c2.z, item.x, itemY, item.z);

            // compute the new inverse matrix (note: shortcut use identity because aligned cubes normals aren't affected by any non uniform scale
            _sysmemBuffer[(windowOffsetInFloat4 + _maxInstancePerWindow * 3 * 1 + index * 3 + 0)] = new float4(rot.c0.x, rot.c1.x, rot.c2.x, rot.c0.y);
            _sysmemBuffer[(windowOffsetInFloat4 + _maxInstancePerWindow * 3 * 1 + index * 3 + 1)] = new float4(rot.c1.y, rot.c2.y, rot.c0.z, rot.c1.z);
            _sysmemBuffer[(windowOffsetInFloat4 + _maxInstancePerWindow * 3 * 1 + index * 3 + 2)] = new float4(rot.c2.z, -item.x, -itemY, -item.z);

            // update colors
            _sysmemBuffer[windowOffsetInFloat4 + _maxInstancePerWindow * 3 * 2 + index] = item.color;

            item.z = item.z + _dt * _speed;
            if (item.z > 100)
            {
                item.z = -5;
            }

            backgroundItems[index] = item;
        }
    }

    [BurstCompile]
    JobHandle UpdatePositions(float dt, float speed, JobHandle jobFence)
    {
        int totalGpuBufferSize;
        int alignedWindowSize;
        NativeArray<float4> sysmemBuffer = m_brgContainer.GetSysmemBuffer(out totalGpuBufferSize, out alignedWindowSize);

        UpdatePositionsJob myJob = new UpdatePositionsJob()
        {
            _sysmemBuffer = sysmemBuffer,
            backgroundItems = m_backgroundItems,
            _maxInstancePerWindow = alignedWindowSize / kGpuItemSize,
            _windowSizeInFloat4 = alignedWindowSize / 16,
            _dt = dt,
            _speed = speed
        };
        jobFence = myJob.ScheduleParallel(m_backgroundItems.Length, 4, jobFence); // 4 slices per job
        return jobFence;
    }

    private float lastTime = 0;
    private float now = 1;

    void Update()
    {
        JobHandle jobFence = new JobHandle();

        now = Time.time;
        m_updateJobFence = UpdatePositions(Time.deltaTime, speed, jobFence);
    }

    private void LateUpdate()
    {
        m_updateJobFence.Complete();
        m_brgContainer.UploadGpuData(bulletCount);
    }

    private void OnDestroy()
    {
        if (m_brgContainer != null)
            m_brgContainer.Shutdown();
        m_backgroundItems.Dispose();
    }
}