using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;
using Unity.Mathematics;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;

/*
    每个实例的数据如下，有7个float4
        - obj2world matrix ( 3 * float4 )
        - world2obj matrix ( 3 * float4 )
        - color ( 1 * float4 )

    Do not forget data is stored in SoA

*/


public unsafe class BRG_Container
{
    // In GLES mode, BRG raw buffer is a constant buffer (UBO)
    private bool UseConstantBuffer => BatchRendererGroup.BufferTarget == BatchBufferTarget.ConstantBuffer;
    private bool m_castShadows;


    private int m_maxInstances; // 实例数 maximum item in this container
    private int m_instanceCount; // 当前实例数 current item count
    private int m_alignedGPUWindowSize; // 原始window尺寸 BRG raw window size
    private int m_maxInstancePerWindow; // 一个window的实例数 max instance per window (
    private int m_windowCount; // window数量 amount of window (1 in SSBO mode, n in UBO mode)
    private int m_totalGpuBufferSize; // buffer的总大小，就是window数*每个window尺寸 total size of the raw buffer
    private NativeArray<float4> m_sysmemBuffer; // 传给GraphicsBuffer的数据 system memory copy of the raw buffer
    private bool m_initialized;
    private int m_instanceSize; // 每个实例的字节数 item size, in bytes
    private BatchID[] m_batchIDs; // one batchID per window
    private BatchMaterialID m_materialID;
    private BatchMeshID m_meshID;
    private BatchRendererGroup m_BatchRendererGroup; // BRG 
    private GraphicsBuffer m_GPUPersistentInstanceData; // GPU raw buffer (could be SSBO or UBO)

    /// <param name="maxInstances">实例数</param>
    /// <param name="instanceSize">每个实例字节数</param>
    /// <returns></returns>
    public bool Init(Mesh mesh, Material mat, int maxInstances, int instanceSize, bool castShadows)
    {
        Debug.Log("UseConstantBuffer:" + UseConstantBuffer);
        m_BatchRendererGroup = new BatchRendererGroup(this.OnPerformCulling, IntPtr.Zero);

        m_instanceSize = instanceSize;
        m_instanceCount = 0;
        m_maxInstances = maxInstances;
        m_castShadows = castShadows;

        // BRG uses a large GPU buffer. This is a RAW buffer on almost all platforms, and a constant buffer on GLES
        // In case of constant buffer, we split it into several "windows" of BatchRendererGroup.GetConstantBufferMaxWindowSize() bytes each
        // 这里跑的为false
        if (UseConstantBuffer)
        {
            m_alignedGPUWindowSize = BatchRendererGroup.GetConstantBufferMaxWindowSize();
            m_maxInstancePerWindow = m_alignedGPUWindowSize / instanceSize;
            m_windowCount = (m_maxInstances + m_maxInstancePerWindow - 1) / m_maxInstancePerWindow;
            m_totalGpuBufferSize = m_windowCount * m_alignedGPUWindowSize;
            m_GPUPersistentInstanceData = new GraphicsBuffer(GraphicsBuffer.Target.Constant, m_totalGpuBufferSize / 16, 16);
        }
        else
        {
            m_alignedGPUWindowSize = (m_maxInstances * instanceSize + 15) & (-16);
            m_maxInstancePerWindow = maxInstances;
            m_windowCount = 1;
            m_totalGpuBufferSize = m_windowCount * m_alignedGPUWindowSize;
            m_GPUPersistentInstanceData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, m_totalGpuBufferSize / 4, 4);
        }

        //需要处理的属性有3个，obj2world, world2obj 和 baseColor
        var batchMetadata = new NativeArray<MetadataValue>(3, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

        int objectToWorldID = Shader.PropertyToID("unity_ObjectToWorld");
        int worldToObjectID = Shader.PropertyToID("unity_WorldToObject");
        int colorID = Shader.PropertyToID("_BaseColor");

        //要传给GraphicBuffer的数据
        m_sysmemBuffer = new NativeArray<float4>(m_totalGpuBufferSize / 16, Allocator.Persistent, NativeArrayOptions.ClearMemory);

        // register one kind of batch per "window" in the large BRG raw buffer
        //注册数据
        m_batchIDs = new BatchID[m_windowCount];
        for (int b = 0; b < m_windowCount; b++)
        {
            batchMetadata[0] = CreateMetadataValue(objectToWorldID, 0, true); // matrices
            batchMetadata[1] = CreateMetadataValue(worldToObjectID, m_maxInstancePerWindow * 3 * 16, true); // 16是4个float，4*3的矩阵就是3*16
            batchMetadata[2] = CreateMetadataValue(colorID, m_maxInstancePerWindow * 3 * 2 * 16, true); // colors，offset为前面两个4*3矩阵
            int offset = b * m_alignedGPUWindowSize;
            //AddBatch，让BRG关联上了metadata和GraphicBuffer，之后每帧更新数据给GraphicBuffer.SetData就可以更新数据
            m_batchIDs[b] = m_BatchRendererGroup.AddBatch(batchMetadata, m_GPUPersistentInstanceData.bufferHandle, (uint) offset, UseConstantBuffer ? (uint) m_alignedGPUWindowSize : 0);
        }

        //不需要这个metadata描述了
        batchMetadata.Dispose();

        //设置一个很大的包围盒，确保BRG不会被剔除
        UnityEngine.Bounds bounds = new Bounds(new Vector3(0, 0, 0), new Vector3(1048576.0f, 1048576.0f, 1048576.0f));
        m_BatchRendererGroup.SetGlobalBounds(bounds);

        // 注册网格和材质
        if (mesh) m_meshID = m_BatchRendererGroup.RegisterMesh(mesh);
        if (mat) m_materialID = m_BatchRendererGroup.RegisterMaterial(mat);

        m_initialized = true;
        return true;
    }

//  根据实例数更新GPU数据
//  Because of SoA and this class is managing 3 BRG properties ( 2 matrices & 1 color ), the last window could use up to 3 SetData
    [BurstCompile]
    public bool UploadGpuData(int instanceCount)
    {
        if ((uint) instanceCount > (uint) m_maxInstances)
            return false;

        m_instanceCount = instanceCount;
        int completeWindows = m_instanceCount / m_maxInstancePerWindow;

        // update all complete windows in one go
        if (completeWindows > 0)
        {
            int sizeInFloat4 = (completeWindows * m_alignedGPUWindowSize) / 16;
            m_GPUPersistentInstanceData.SetData(m_sysmemBuffer, 0, 0, sizeInFloat4);
        }

        // then upload data for the last (incomplete) window
        int lastBatchId = completeWindows;
        int itemInLastBatch = m_instanceCount - m_maxInstancePerWindow * completeWindows;

        if (itemInLastBatch > 0)
        {
            int windowOffsetInFloat4 = (lastBatchId * m_alignedGPUWindowSize) / 16;
            int offsetMat1 = windowOffsetInFloat4 + m_maxInstancePerWindow * 0;
            int offsetMat2 = windowOffsetInFloat4 + m_maxInstancePerWindow * 3;
            int offsetColor = windowOffsetInFloat4 + m_maxInstancePerWindow * 3 * 2;
            m_GPUPersistentInstanceData.SetData(m_sysmemBuffer, offsetMat1, offsetMat1, itemInLastBatch * 3); // 3 float4 for obj2world
            m_GPUPersistentInstanceData.SetData(m_sysmemBuffer, offsetMat2, offsetMat2, itemInLastBatch * 3); // 3 float4 for world2obj
            m_GPUPersistentInstanceData.SetData(m_sysmemBuffer, offsetColor, offsetColor, itemInLastBatch * 1); // 1 float4 for color
        }

        return true;
    }

    // Release all allocated buffers
    public void Shutdown()
    {
        if (m_initialized)
        {
            for (uint b = 0; b < m_windowCount; b++)
                m_BatchRendererGroup.RemoveBatch(m_batchIDs[b]);

            m_BatchRendererGroup.UnregisterMaterial(m_materialID);
            m_BatchRendererGroup.UnregisterMesh(m_meshID);
            m_BatchRendererGroup.Dispose();
            m_GPUPersistentInstanceData.Dispose();
            m_sysmemBuffer.Dispose();
        }
    }

    // return the system memory buffer and the window size, so BRG_Background and BRG_Debris can fill the buffer with new content
    public NativeArray<float4> GetSysmemBuffer(out int totalSize, out int alignedWindowSize)
    {
        totalSize = m_totalGpuBufferSize;
        alignedWindowSize = m_alignedGPUWindowSize;
        return m_sysmemBuffer;
    }

    // helper function to create the 32bits metadata value. Bit 31 means property have different value per instance
    static MetadataValue CreateMetadataValue(int nameID, int gpuOffset, bool isPerInstance)
    {
        const uint kIsPerInstanceBit = 0x80000000;
        return new MetadataValue
        {
            NameID = nameID,
            Value = (uint) gpuOffset | (isPerInstance ? (kIsPerInstanceBit) : 0),
        };
    }

    // Helper function to allocate BRG buffers during the BRG callback function
    private static T* Malloc<T>(uint count) where T : unmanaged
    {
        return (T*) UnsafeUtility.Malloc(
            UnsafeUtility.SizeOf<T>() * count,
            UnsafeUtility.AlignOf<T>(),
            Allocator.TempJob);
    }

    // Main BRG entry point per frame. In this sample we won't use BatchCullingContext as we don't need culling
    // This callback is responsible to fill cullingOutput with all draw commands we need to render all the items
    [BurstCompile]
    public JobHandle OnPerformCulling(BatchRendererGroup rendererGroup, BatchCullingContext cullingContext, BatchCullingOutput cullingOutput, IntPtr userContext)
    {
        if (m_initialized)
        {
            BatchCullingOutputDrawCommands drawCommands = new BatchCullingOutputDrawCommands();

            // calculate the amount of draw commands we need in case of UBO mode (one draw command per window)
            //每个window一个draw command
            int drawCommandCount = (m_instanceCount + m_maxInstancePerWindow - 1) / m_maxInstancePerWindow;
            int maxInstancePerDrawCommand = m_maxInstancePerWindow;
            drawCommands.drawCommandCount = drawCommandCount;

            // Allocate a single BatchDrawRange. ( all our draw commands will refer to this BatchDrawRange)
            drawCommands.drawRangeCount = 1;
            drawCommands.drawRanges = Malloc<BatchDrawRange>(1);
            drawCommands.drawRanges[0] = new BatchDrawRange
            {
                drawCommandsBegin = 0,
                drawCommandsCount = (uint) drawCommandCount,
                filterSettings = new BatchFilterSettings
                {
                    renderingLayerMask = 1,
                    layer = 0,
                    motionMode = MotionVectorGenerationMode.Camera,
                    shadowCastingMode = m_castShadows ? ShadowCastingMode.On : ShadowCastingMode.Off,
                    receiveShadows = true,
                    staticShadowCaster = false,
                    allDepthSorted = false
                }
            };

            if (drawCommands.drawCommandCount > 0)
            {
                // as we don't need culling, the visibility int array buffer will always be {0,1,2,3,...} for each draw command
                // so we just allocate maxInstancePerDrawCommand and fill it
                int visibilityArraySize = maxInstancePerDrawCommand;
                if (m_instanceCount < visibilityArraySize)
                    visibilityArraySize = m_instanceCount;

                drawCommands.visibleInstances = Malloc<int>((uint) visibilityArraySize);

                // As we don't need any frustum culling in our context, we fill the visibility array with {0,1,2,3,...}
                for (int i = 0; i < visibilityArraySize; i++)
                    drawCommands.visibleInstances[i] = i;

                // Allocate the BatchDrawCommand array (drawCommandCount entries)
                // In SSBO mode, drawCommandCount will be just 1
                drawCommands.drawCommands = Malloc<BatchDrawCommand>((uint) drawCommandCount);
                int left = m_instanceCount;
                for (int b = 0; b < drawCommandCount; b++)
                {
                    int inBatchCount = left > maxInstancePerDrawCommand ? maxInstancePerDrawCommand : left;
                    drawCommands.drawCommands[b] = new BatchDrawCommand
                    {
                        visibleOffset = (uint) 0, // all draw command is using the same {0,1,2,3...} visibility int array
                        visibleCount = (uint) inBatchCount,
                        batchID = m_batchIDs[b],
                        materialID = m_materialID,
                        meshID = m_meshID,
                        submeshIndex = 0,
                        splitVisibilityMask = 0xff,
                        flags = BatchDrawCommandFlags.None,
                        sortingPosition = 0
                    };
                    left -= inBatchCount;
                }
            }

            cullingOutput.drawCommands[0] = drawCommands;
            drawCommands.instanceSortingPositions = null;
            drawCommands.instanceSortingPositionFloatCount = 0;
        }

        return new JobHandle();
    }
}