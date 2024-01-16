using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

public class MyBRG : MonoBehaviour
{
    public Mesh m_mesh;
    public Material m_material;
    public int col;
    public int row;
    public int h;
    public float gap = 1.2f;
    public float speed = 10;
    public float changeTime = 1;
    private const int kGpuItemSize = (3 * 2 + 1) * 16; //  每个实例字节数 ( 2 * 4x3 matrices + 1 color per item )

    private BRG_Container m_brgContainer;
    private JobHandle m_updateJobFence;
    private int m_itemCount;

    private struct BackgroundItem
    {
        public float x;
        public float y;
        public float z;
        public Vector4 color;
        public float dir;
    };

    private NativeArray<BackgroundItem> m_backgroundItems;

    void Start()
    {
        m_itemCount = col * row * h;
        m_brgContainer = new BRG_Container();
        m_brgContainer.Init(m_mesh, m_material, m_itemCount, kGpuItemSize, false);

        m_backgroundItems = new NativeArray<BackgroundItem>(m_itemCount, Allocator.Persistent, NativeArrayOptions.ClearMemory);

        InjectNewSlice();

        m_brgContainer.UploadGpuData(m_itemCount);
    }

    [BurstCompile]
    private void InjectNewSlice()
    {
        var index = 0;
        for (int i = 0; i < col; i++)
        {
            for (int j = 0; j < row; j++)
            {
                for (int k = 0; k < h; k++)
                {
                    float x = i * gap;
                    float z = j * gap;
                    float y = k * gap;
                    BackgroundItem item = new BackgroundItem();
                    item.x = x;
                    item.y = y;
                    item.z = z;
                    item.color = new Vector4(Random.Range(0.0f, 1.0f), Random.Range(0.0f, 1.0f), Random.Range(0.0f, 1.0f), 1.0f);
                    item.dir = j % 2 == 0 ? 1 : -1;
                    m_backgroundItems[index] = item;
                    index++;
                }
            }
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


            int windowId = 0; //System.Math.DivRem(slice * backgroundW + x, _maxInstancePerWindow, out i);
            int windowOffsetInFloat4 = windowId * _windowSizeInFloat4;

            // compute the new current frame matrix
            _sysmemBuffer[(windowOffsetInFloat4 + index * 3 + 0)] = new float4(1, 0, 0, 0);
            _sysmemBuffer[(windowOffsetInFloat4 + index * 3 + 1)] = new float4(1, 0, 0, 0);
            _sysmemBuffer[(windowOffsetInFloat4 + index * 3 + 2)] = new float4(1, item.x, item.y, item.z);

            // compute the new inverse matrix (note: shortcut use identity because aligned cubes normals aren't affected by any non uniform scale
            _sysmemBuffer[(windowOffsetInFloat4 + _maxInstancePerWindow * 3 * 1 + index * 3 + 0)] = new float4(1, 0, 0, 0);
            _sysmemBuffer[(windowOffsetInFloat4 + _maxInstancePerWindow * 3 * 1 + index * 3 + 1)] = new float4(1, 0, 0, 0);
            _sysmemBuffer[(windowOffsetInFloat4 + _maxInstancePerWindow * 3 * 1 + index * 3 + 2)] = new float4(1, 0, 0, 0);

            // update colors
            _sysmemBuffer[windowOffsetInFloat4 + _maxInstancePerWindow * 3 * 2 + index] = item.color;

            item.y = item.y + _dt * item.dir * _speed;
            if (_change)
            {
                item.dir *= -1;
            }

            backgroundItems[index] = item;
        }
    }

    [BurstCompile]
    JobHandle UpdatePositions(bool change, float dt, float speed, JobHandle jobFence)
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
            _change = change,
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
        bool change = false;
        if (now - lastTime > changeTime)
        {
            lastTime = now;
            change = true;
        }
        now = Time.time;
        m_updateJobFence = UpdatePositions(change, Time.deltaTime, speed, jobFence);
    }

    private void LateUpdate()
    {
        m_updateJobFence.Complete();
        m_brgContainer.UploadGpuData(m_itemCount);
    }

    private void OnDestroy()
    {
        if (m_brgContainer != null)
            m_brgContainer.Shutdown();
        m_backgroundItems.Dispose();
    }
}