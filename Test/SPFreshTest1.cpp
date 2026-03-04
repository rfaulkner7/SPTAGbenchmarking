// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/SPANNResultIterator.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/DiskIO.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Test.h"
#include "inc/TestDataGenerator.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace SPTAG;

namespace SPFreshTest
{
SizeType N = 10000;
DimensionType M = 128;
int K = 10;
int queries = 100;

// Build configuration options
struct BuildConfig {
    int selectHeadThreads = 16;
    int buildHeadThreads = 16;
    int buildSSDThreads = 16;
    bool parallelBKTBuild = false;
    int bktKmeansK = 32;
    int bktLeafSize = 8;
    float bktLambdaFactor = -1.0f;
    double ratio = 0.2;
    int selectThreshold = 0;
    int splitFactor = 0;
    int splitThreshold = 0;
};

template <typename T>
std::shared_ptr<VectorIndex> BuildIndex(const std::string &outDirectory, std::shared_ptr<VectorSet> vecset,
                                        std::shared_ptr<MetadataSet> metaset, const std::string &distMethod = "L2",
                                        const BuildConfig &buildConfig = BuildConfig())
{
    auto vecIndex = VectorIndex::CreateInstance(IndexAlgoType::SPANN, GetEnumValueType<T>());

    std::string configuration = R"(
        [Base]
            DistCalcMethod=L2
            IndexAlgoType=BKT
            ValueType=)" + Helper::Convert::ConvertToString(GetEnumValueType<T>()) +
                                R"(
            Dim=)" + std::to_string(M) +
                                R"(
            IndexDirectory=)" + outDirectory +
                                R"(

        [SelectHead]
            isExecute=true
            NumberOfThreads=)" + std::to_string(buildConfig.selectHeadThreads) +
                                R"(
            SelectThreshold=)" + std::to_string(buildConfig.selectThreshold) +
                                R"(
            SplitFactor=)" + std::to_string(buildConfig.splitFactor) +
                                R"(
            SplitThreshold=)" + std::to_string(buildConfig.splitThreshold) +
                                R"(
            Ratio=)" + std::to_string(buildConfig.ratio) +
                                R"(
            ParallelBKTBuild=)" + (buildConfig.parallelBKTBuild ? "true" : "false") +
                                R"(
            BKTKmeansK=)" + std::to_string(buildConfig.bktKmeansK) +
                                R"(
            BKTLeafSize=)" + std::to_string(buildConfig.bktLeafSize) +
                                R"(
            BKTLambdaFactor=)" + std::to_string(buildConfig.bktLambdaFactor) +
                                R"(

        [BuildHead]
            isExecute=true
            ParallelBKTBuild=)" + (buildConfig.parallelBKTBuild ? "true" : "false") + R"(
            NumberOfThreads=)" + std::to_string(buildConfig.buildHeadThreads) +
                                R"(

        [BuildSSDIndex]
            isExecute=true
            BuildSsdIndex=true
            InternalResultNum=64
            SearchInternalResultNum=64
            NumberOfThreads=)" + std::to_string(buildConfig.buildSSDThreads) +
                                R"(
	    PostingPageLimit=)" + std::to_string(4 * sizeof(T)) +
                                R"(
            SearchPostingPageLimit=)" +
                                std::to_string(4 * sizeof(T)) + R"(
            TmpDir=tmpdir
            Storage=FILEIO
            SpdkBatchSize=64
            ExcludeHead=false
            ResultNum=10
            SearchThreadNum=2
            Update=true
            SteadyState=true
            InsertThreadNum=1
            AppendThreadNum=1
            ReassignThreadNum=0
            DisableReassign=false
            ReassignK=64
            LatencyLimit=50.0
            SearchDuringUpdate=true
            MergeThreshold=10
            Sampling=4
            BufferLength=6
            InPlace=true
            StartFileSizeGB=1
            OneClusterCutMax=true
            ConsistencyCheck=true
            ChecksumCheck=true
            ChecksumInRead=false
            DeletePercentageForRefine=0.4
            AsyncAppendQueueSize=0
            AllowZeroReplica=false
        )";

    std::shared_ptr<Helper::DiskIO> buffer(new Helper::SimpleBufferIO());
    Helper::IniReader reader;
    if (!buffer->Initialize(configuration.data(), std::ios::in, configuration.size()))
        return nullptr;
    if (ErrorCode::Success != reader.LoadIni(buffer))
        return nullptr;

    std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"};
    for (const auto &sec : sections)
    {
        auto params = reader.GetParameters(sec.c_str());
        for (const auto &[key, val] : params)
        {
            vecIndex->SetParameter(key.c_str(), val.c_str(), sec.c_str());
        }
    }

    auto buildStatus = vecIndex->BuildIndex(vecset, metaset, true, false, false);
    if (buildStatus != ErrorCode::Success)
        return nullptr;

    return vecIndex;
}

template <typename T>
std::vector<QueryResult> SearchOnly(std::shared_ptr<VectorIndex> &vecIndex, std::shared_ptr<VectorSet> &queryset, int k)
{
    std::vector<QueryResult> res(queryset->Count(), QueryResult(nullptr, k, true));

    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i < queryset->Count(); i++)
    {
        res[i].SetTarget(queryset->GetVector(i));
        vecIndex->SearchIndex(res[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    float avgUs =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / static_cast<float>(queryset->Count());
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Avg search time: %.2fus/query\n", avgUs);

    return res;
}

template <typename T>
float EvaluateRecall(const std::vector<QueryResult> &res, std::shared_ptr<VectorIndex> &vecIndex,
                     std::shared_ptr<VectorSet> &queryset, std::shared_ptr<VectorSet> &truth,
                     std::shared_ptr<VectorSet> &baseVec, std::shared_ptr<VectorSet> &addVec, SizeType baseCount, int k,
                     int batch)
{
    if (!truth)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Truth data is null. Cannot compute recall.\n");
        return 0.0f;
    }

    const SizeType recallK = min(k, static_cast<int>(truth->Dimension()));
    float totalRecall = 0.0f;
    float eps = 1e-4f;

    for (size_t i = 0; i < res.size(); ++i)
    {
        const SizeType *truthNN = reinterpret_cast<const SizeType *>(truth->GetVector(i + batch * queryset->Count()));
        for (int j = 0; j < recallK; ++j)
        {
            SizeType truthVid = truthNN[j];
            float truthDist =
                (truthVid < baseCount)
                    ? vecIndex->ComputeDistance(queryset->GetVector(i), baseVec->GetVector(truthVid))
                    : vecIndex->ComputeDistance(queryset->GetVector(i), addVec->GetVector(truthVid - baseCount));

            for (int l = 0; l < k; ++l)
            {
                const auto result = res[i].GetResult(l);
                if (truthVid == result->VID ||
                    std::fabs(truthDist - result->Dist) <= eps * (std::fabs(truthDist) + eps))
                {
                    totalRecall += 1.0f;
                    break;
                }
            }
        }
    }

    float avgRecall = totalRecall / (res.size() * recallK);
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recall %d@%d = %.4f\n", k, recallK, avgRecall);
    return avgRecall;
}

template <typename T>
float Search(std::shared_ptr<VectorIndex> &vecIndex, std::shared_ptr<VectorSet> &queryset,
             std::shared_ptr<VectorSet> &baseVec, std::shared_ptr<VectorSet> &addVec, int k,
             std::shared_ptr<VectorSet> &truth, SizeType baseCount, int batch = 0)
{
    auto results = SearchOnly<T>(vecIndex, queryset, k);
    return EvaluateRecall<T>(results, vecIndex, queryset, truth, baseVec, addVec, baseCount, k, batch);
}

template <typename ValueType>
void InsertVectors(SPANN::Index<ValueType> *p_index, int insertThreads, int step, std::shared_ptr<VectorSet> addset,
                   std::shared_ptr<MetadataSet> &metaset, int start = 0)
{
    SPANN::Options &p_opts = *(p_index->GetOptions());
    p_index->ForceCompaction();
    p_index->GetDBStat();

    std::vector<std::thread> threads;

    std::atomic_size_t vectorsSent(start);
    auto func = [&]() {
        size_t index = start;
        while (true)
        {
            index = vectorsSent.fetch_add(1);
            if (index < start + step)
            {
                if ((index & ((1 << 5) - 1)) == 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", (index - start) * 100.0 / step);
                }
                ByteArray p_meta = metaset->GetMetadata((SizeType)index);
                std::uint64_t *offsets = new std::uint64_t[2]{0, p_meta.Length()};
                std::shared_ptr<MetadataSet> meta(
                    new MemMetadataSet(p_meta, ByteArray((std::uint8_t *)offsets, 2 * sizeof(std::uint64_t), true), 1));
                auto status = p_index->AddIndex(addset->GetVector((SizeType)index), 1, p_opts.m_dim, meta, true);
                BOOST_REQUIRE(status == ErrorCode::Success);
            }
            else
            {
                return;
            }
        }
    };
    for (int j = 0; j < insertThreads; j++)
    {
        threads.emplace_back(func);
    }
    for (auto &thread : threads)
    {
        thread.join();
    }

    while (!p_index->AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

template <typename T>
void LoadTestData(const std::string &vectorPath, const std::string &queryPath, const std::string &truthPath,
                  std::shared_ptr<SPTAG::VectorSet> &vecset, std::shared_ptr<SPTAG::MetadataSet> &metaset,
                  std::shared_ptr<SPTAG::VectorSet> &addvecset, std::shared_ptr<SPTAG::MetadataSet> &addmetaset,
                  std::shared_ptr<SPTAG::VectorSet> &queryset, int dimension, int k, std::string distMethodStr,
                  int base, int batchinsert, int batchdelete, int batches, std::shared_ptr<SPTAG::VectorSet> &truths,
                  bool normalize)
{
    VectorFileType fileType = VectorFileType::DEFAULT;
    if (vectorPath.find(".fvecs") != std::string::npos || vectorPath.find(".ivecs") != std::string::npos)
    {
        fileType = VectorFileType::XVEC;
    }
    auto vectorOptions =
        std::shared_ptr<Helper::ReaderOptions>(new Helper::ReaderOptions(GetEnumValueType<T>(), dimension, fileType));

    // Load dataset using VectorSetReader
    BOOST_TEST_MESSAGE("Loading vectors from: " << vectorPath);
    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
    auto error = vectorReader->LoadFile(vectorPath);
    if (ErrorCode::Success != error)
    {
        BOOST_FAIL("Failed to load vector file: " << vectorPath << "Error: " << static_cast<uint64_t>(error));        
        return;
    }
    auto allVectors = vectorReader->GetVectorSet();
    int totalVectors = allVectors->Count();

    // Load queries
    BOOST_TEST_MESSAGE("Loading queries from: " << queryPath);
    auto queryReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
    if (ErrorCode::Success != queryReader->LoadFile(queryPath))
    {
        BOOST_FAIL("Failed to load query file: " << queryPath);
        return;
    }
    queryset = queryReader->GetVectorSet();
    int totalQueries = queryset->Count();

    BOOST_TEST_MESSAGE("Loaded " << totalVectors << " vectors" << ", " << totalQueries << " queries.");
    BOOST_REQUIRE(totalVectors >= base + batchinsert * batches);

    // Split into base (for building index) and insert (for benchmarking)
    ByteArray baseData = ByteArray::Alloc(sizeof(T) * base * dimension);
    for (int i = 0; i < base; i++)
    {
        memcpy(baseData.Data() + i * sizeof(T) * dimension, allVectors->GetVector(i), sizeof(T) * dimension);
    }
    vecset = std::make_shared<BasicVectorSet>(baseData, GetEnumValueType<T>(), dimension, base);
    metaset = TestUtils::TestDataGenerator<T>::GenerateMetadataSet(base, 0);

    ByteArray insertData = ByteArray::Alloc(sizeof(T) * (batchinsert * batches) * dimension);
    for (int i = 0; i < batchinsert * batches; i++)
    {
        memcpy(insertData.Data() + i * sizeof(T) * dimension, allVectors->GetVector(base + i), sizeof(T) * dimension);
    }
    addvecset = std::make_shared<BasicVectorSet>(insertData, GetEnumValueType<T>(), dimension, batchinsert * batches);
    addmetaset = TestUtils::TestDataGenerator<T>::GenerateMetadataSet(batchinsert * batches, base);

    ByteArray allData = ByteArray::Alloc(sizeof(T) * (vecset->Count() + addvecset->Count()) * dimension);
    memcpy(allData.Data(), vecset->GetData(), sizeof(T) * vecset->Count() * dimension);
    memcpy(allData.Data() + sizeof(T) * vecset->Count() * dimension, addvecset->GetData(),
           sizeof(T) * addvecset->Count() * dimension);
    auto allvecset = std::make_shared<BasicVectorSet>(allData, GetEnumValueType<T>(), dimension,
                                                      vecset->Count() + addvecset->Count());

    // Generate ground truth
    std::string truthfile = truthPath + "_convertTruth";
    if (fileexists(truthfile.c_str()))
    {
        auto opts =
            std::make_shared<Helper::ReaderOptions>(GetEnumValueType<float>(), dimension, VectorFileType::DEFAULT);
        auto reader = Helper::VectorSetReader::CreateInstance(opts);
        if (ErrorCode::Success != reader->LoadFile(truthfile))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file %s\n", truthfile.c_str());
            exit(1);
        }
        truths = reader->GetVectorSet();
        return;
    }

    DistCalcMethod distMethod;
    Helper::Convert::ConvertStringTo(distMethodStr.c_str(), distMethod);
    if (normalize && distMethod == DistCalcMethod::Cosine)
    {
        COMMON::Utils::BatchNormalize((T *)allvecset->GetData(), allvecset->Count(), allvecset->Dimension(),
                                      COMMON::Utils::GetBase<T>(), 5);
    }

    ByteArray tru = ByteArray::Alloc(sizeof(float) * (batches + 1) * queryset->Count() * k);
    int start = 0;
    int end = base;
    int maxthreads = std::thread::hardware_concurrency();
    for (int iter = 0; iter < batches + 1; iter++)
    {
        std::vector<std::thread> mythreads;
        mythreads.reserve(maxthreads);
        std::atomic_size_t sent(0);
        for (int tid = 0; tid < maxthreads; tid++)
        {
            mythreads.emplace_back([&, tid]() {
                size_t i = 0;
                while (true)
                {
                    i = sent.fetch_add(1);
                    if (i < queryset->Count())
                    {
                        SizeType *neighbors = ((SizeType *)tru.Data()) + iter * (queryset->Count() * k) + i * k;
                        COMMON::QueryResultSet<T> res((const T *)queryset->GetVector(i), k);
                        for (SizeType j = start; j < end; ++j)
                        {
                            float dist = COMMON::DistanceUtils::ComputeDistance(
                                res.GetTarget(), reinterpret_cast<T *>(allvecset->GetVector(j)), dimension, distMethod);
                            res.AddPoint(j, dist);
                        }
                        res.SortResult();
                        for (int j = 0; j < k; ++j)
                            neighbors[j] = res.GetResult(j)->VID;
                    }
                    else
                    {
                        return;
                    }
                }
            });
        }
        for (auto &t : mythreads)
        {
            t.join();
        }
        mythreads.clear();
        start += batchdelete;
        end += batchinsert;
    }
    truths = std::make_shared<BasicVectorSet>(tru, GetEnumValueType<float>(), k, (batches + 1) * queryset->Count());
    truths->Save(truthfile);
}

template <typename T>
void RunBenchmark(const std::string &vectorPath, const std::string &queryPath, const std::string &truthPath,
                  const std::string &indexPath, int dimension, int baseVectorCount, int insertVectorCount,
                  int batchSize, int topK, int numThreads, int numQueries,
                  const std::string &outputFile = "output.json",
                  const BuildConfig &buildConfig = BuildConfig())
{
    M = dimension;

    // Variables to collect JSON output data
    std::ostringstream benchmark1Data, benchmark2Data, benchmark3Data, benchmark4Data, benchmark5Data, benchmark6Data,
        benchmark7Data, benchmark8Data;

    // Generate test data
    std::shared_ptr<VectorSet> vecset, addvecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;
    LoadTestData<T>(vectorPath, queryPath, truthPath, vecset, metaset, addvecset, addmetaset, queryset, dimension, topK,
                    "L2", baseVectorCount, insertVectorCount, 0, 1, truth, false);

    //// Build initial index, unless index exists then just load index
    //BOOST_TEST_MESSAGE("\n=== Building Index ===");
    //std::shared_ptr<VectorIndex> index = BuildIndex<T>(indexPath, vecset, metaset);
    //BOOST_REQUIRE(index != nullptr);

    //BOOST_TEST_MESSAGE("Index built successfully with " << baseVectorCount << " vectors");
    // Build or load index
    BOOST_TEST_MESSAGE("\n=== Preparing Index ===");

    std::shared_ptr<VectorIndex> index;

    if (std::filesystem::exists(indexPath.c_str()) && std::filesystem::is_directory(indexPath.c_str()) &&
        !std::filesystem::is_empty(indexPath.c_str()))
    {
        BOOST_TEST_MESSAGE("Index exists. Loading from: " << indexPath);

        ErrorCode loadResult = VectorIndex::LoadIndex(indexPath + "_saved", index);

        BOOST_REQUIRE(loadResult == ErrorCode::Success);

        BOOST_TEST_MESSAGE("Index loaded successfully with " << index->GetNumSamples() << " vectors");
    }
    else
    {
        BOOST_TEST_MESSAGE("Index does not exist. Building new index...");

        index = BuildIndex<T>(indexPath, vecset, metaset, "L2", buildConfig);
        BOOST_REQUIRE(index != nullptr);

        BOOST_TEST_MESSAGE("Index built successfully with " << baseVectorCount << " vectors");
    }

    // Benchmark 1: Insert performance
    BOOST_TEST_MESSAGE("\n=== Benchmark 1: Insert Performance ===");
    {
        auto start = std::chrono::high_resolution_clock::now();
        InsertVectors<T>(static_cast<SPANN::Index<T> *>(index.get()), numThreads, insertVectorCount, addvecset,
                         addmetaset, 0);

        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
        double throughput = insertVectorCount / seconds;

        BOOST_TEST_MESSAGE("  Inserted: " << insertVectorCount << " vectors");
        BOOST_TEST_MESSAGE("  Time: " << seconds << " seconds");
        BOOST_TEST_MESSAGE("  Throughput: " << throughput << " vectors/sec");

        // Collect JSON data for Benchmark 1
        benchmark1Data << std::fixed << std::setprecision(4);
        benchmark1Data << "{\n";
        benchmark1Data << "      \"inserted\": " << insertVectorCount << ",\n";
        benchmark1Data << "      \"timeSeconds\": " << seconds << ",\n";
        benchmark1Data << "      \"throughput\": " << throughput << "\n";
        benchmark1Data << "    }";
    }

    // Benchmark 2: Query performance with detailed latency stats
    BOOST_TEST_MESSAGE("\n=== Benchmark 2: Query Performance ===");
    {
        std::vector<float> latencies(numQueries);
        std::atomic_size_t queriesSent(0);
        std::vector<QueryResult> results(numQueries);

        for (int i = 0; i < numQueries; i++)
        {
            results[i] = QueryResult((const T *)queryset->GetVector(i), topK, false);
        }

        std::vector<std::thread> threads;
        threads.reserve(numThreads);

        auto batchStart = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numThreads; i++)
        {
            threads.emplace_back([&]() {
                size_t qid;
                while ((qid = queriesSent.fetch_add(1)) < numQueries)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    index->SearchIndex(results[qid]);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    latencies[qid] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
                }
            });
        }

        for (auto &thread : threads)
            thread.join();

        auto batchEnd = std::chrono::high_resolution_clock::now();
        float batchLatency =
            std::chrono::duration_cast<std::chrono::microseconds>(batchEnd - batchStart).count() / 1000000.0f;

        // Calculate statistics
        float mean = 0, minLat = (std::numeric_limits<float>::max)(), maxLat = 0;
        for (int i = 0; i < numQueries; i++)
        {
            mean += latencies[i];
            minLat = (std::min)(minLat, latencies[i]);
            maxLat = (std::max)(maxLat, latencies[i]);
        }
        mean /= numQueries;

        std::sort(latencies.begin(), latencies.end());
        float p50 = latencies[static_cast<size_t>(numQueries * 0.50)];
        float p90 = latencies[static_cast<size_t>(numQueries * 0.90)];
        float p95 = latencies[static_cast<size_t>(numQueries * 0.95)];
        float p99 = latencies[static_cast<size_t>(numQueries * 0.99)];
        float qps = numQueries / batchLatency;

        BOOST_TEST_MESSAGE("  Queries: " << numQueries);
        BOOST_TEST_MESSAGE("  Mean Latency: " << mean << " ms");
        BOOST_TEST_MESSAGE("  P50 Latency:  " << p50 << " ms");
        BOOST_TEST_MESSAGE("  P90 Latency:  " << p90 << " ms");
        BOOST_TEST_MESSAGE("  P95 Latency:  " << p95 << " ms");
        BOOST_TEST_MESSAGE("  P99 Latency:  " << p99 << " ms");
        BOOST_TEST_MESSAGE("  Min Latency:  " << minLat << " ms");
        BOOST_TEST_MESSAGE("  Max Latency:  " << maxLat << " ms");
        BOOST_TEST_MESSAGE("  QPS:          " << qps);

        // Collect JSON data for Benchmark 2
        benchmark2Data << std::fixed << std::setprecision(4);
        benchmark2Data << "{\n";
        benchmark2Data << "      \"numQueries\": " << numQueries << ",\n";
        benchmark2Data << "      \"meanLatency\": " << mean << ",\n";
        benchmark2Data << "      \"p50\": " << p50 << ",\n";
        benchmark2Data << "      \"p90\": " << p90 << ",\n";
        benchmark2Data << "      \"p95\": " << p95 << ",\n";
        benchmark2Data << "      \"p99\": " << p99 << ",\n";
        benchmark2Data << "      \"minLatency\": " << minLat << ",\n";
        benchmark2Data << "      \"maxLatency\": " << maxLat << ",\n";
        benchmark2Data << "      \"qps\": " << qps << "\n";
        benchmark2Data << "    }";
    }

    // Benchmark 3: Save index to disk
    BOOST_TEST_MESSAGE("\n=== Benchmark 3: Save Index ===");
    {
        auto start = std::chrono::high_resolution_clock::now();
        BOOST_REQUIRE(ErrorCode::Success == index->SaveIndex(indexPath + "_saved"));
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
        BOOST_TEST_MESSAGE("  Save Time: " << seconds << " seconds");
        BOOST_TEST_MESSAGE("  Save completed successfully");

        // Collect JSON data for Benchmark 3
        benchmark3Data << std::fixed << std::setprecision(4);
        benchmark3Data << "{\n";
        benchmark3Data << "      \"timeSeconds\": " << seconds << "\n";
        benchmark3Data << "    }";
    }

    BOOST_TEST_MESSAGE("Starting Benchmark 4...");

    // Benchmark 4: Clean up memory and reload from disk
    BOOST_TEST_MESSAGE("\n=== Benchmark 4: Memory Cleanup & Reload ===");

    // Explicitly clean up memory
    BOOST_TEST_MESSAGE("Releasing index from memory...");
    index = nullptr;
    BOOST_TEST_MESSAGE("Memory cleanup complete");

    // Reload index
    BOOST_TEST_MESSAGE("Starting to reload index from: " << indexPath + "_saved");
    {
        auto start = std::chrono::high_resolution_clock::now();
        ErrorCode loadResult = VectorIndex::LoadIndex(indexPath + "_saved", index);
        BOOST_TEST_MESSAGE("LoadIndex returned with code: " << (int)loadResult);
        BOOST_REQUIRE(loadResult == ErrorCode::Success);
        auto end = std::chrono::high_resolution_clock::now();

        BOOST_REQUIRE(index != nullptr);

        double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
        int vectorCount = index->GetNumSamples();
        BOOST_TEST_MESSAGE("  Load Time: " << seconds << " seconds");
        BOOST_TEST_MESSAGE("  Index vectors after reload: " << vectorCount);

        // Collect JSON data for Benchmark 4
        benchmark4Data << std::fixed << std::setprecision(4);
        benchmark4Data << "{\n";
        benchmark4Data << "      \"timeSeconds\": " << seconds << ",\n";
        benchmark4Data << "      \"vectorCount\": " << vectorCount << "\n";
        benchmark4Data << "    }";
    }

    BOOST_TEST_MESSAGE("Benchmark 4 completed successfully");

    std::vector<QueryResult> results(numQueries);
    // Benchmark 5: Query performance after reload
    BOOST_TEST_MESSAGE("\n=== Benchmark 5: Query After Reload ===");
    {
        std::vector<float> latencies(numQueries);
        std::atomic_size_t queriesSent(0);

        for (int i = 0; i < numQueries; i++)
        {
            results[i] = QueryResult((const T *)queryset->GetVector(i), topK, false);
        }

        std::vector<std::thread> threads;
        threads.reserve(numThreads);

        auto batchStart = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numThreads; i++)
        {
            threads.emplace_back([&]() {
                size_t qid;
                while ((qid = queriesSent.fetch_add(1)) < numQueries)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    index->SearchIndex(results[qid]);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    latencies[qid] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
                }
            });
        }

        for (auto &thread : threads)
            thread.join();

        auto batchEnd = std::chrono::high_resolution_clock::now();
        float batchLatency =
            std::chrono::duration_cast<std::chrono::microseconds>(batchEnd - batchStart).count() / 1000000.0f;

        float mean = 0;
        for (int i = 0; i < numQueries; i++)
        {
            mean += latencies[i];
        }
        mean /= numQueries;

        std::sort(latencies.begin(), latencies.end());
        float p95 = latencies[static_cast<size_t>(numQueries * 0.95)];
        float p99 = latencies[static_cast<size_t>(numQueries * 0.99)];
        float qps = numQueries / batchLatency;

        BOOST_TEST_MESSAGE("  Mean Latency: " << mean << " ms");
        BOOST_TEST_MESSAGE("  P95 Latency:  " << p95 << " ms");
        BOOST_TEST_MESSAGE("  P99 Latency:  " << p99 << " ms");
        BOOST_TEST_MESSAGE("  QPS:          " << qps);

        // Collect JSON data for Benchmark 5
        benchmark5Data << std::fixed << std::setprecision(4);
        benchmark5Data << "{\n";
        benchmark5Data << "      \"meanLatency\": " << mean << ",\n";
        benchmark5Data << "      \"p95\": " << p95 << ",\n";
        benchmark5Data << "      \"p99\": " << p99 << ",\n";
        benchmark5Data << "      \"qps\": " << qps << "\n";
        benchmark5Data << "    }";
    }

    // Benchmark 6: Recall evaluation (if truth file provided)
    BOOST_TEST_MESSAGE("Checking for truth file: " << truthPath);
    float avgRecall = EvaluateRecall<T>(results, index, queryset, truth, vecset, addvecset, baseVectorCount, topK, 1);
    BOOST_TEST_MESSAGE("  Recall@" << topK << " = " << (avgRecall * 100.0f) << "%");
    BOOST_TEST_MESSAGE("  (Evaluated on " << numQueries << " queries against base vectors)");

    // Collect JSON data for Benchmark 6
    benchmark6Data << std::fixed << std::setprecision(4);
    benchmark6Data << "{\n";
    benchmark6Data << "      \"recallAtK\": " << avgRecall << ",\n";
    benchmark6Data << "      \"k\": " << topK << ",\n";
    benchmark6Data << "      \"numQueries\": " << numQueries << "\n";
    benchmark6Data << "    }";

    BOOST_TEST_MESSAGE("\n=== Benchmark 7: Query again After Reload ===");
    {
        std::vector<float> latencies(numQueries);
        std::atomic_size_t queriesSent(0);

        for (int i = 0; i < numQueries; i++)
        {
            results[i] = QueryResult((const T *)queryset->GetVector(i), topK, false);
        }

        std::vector<std::thread> threads;
        threads.reserve(numThreads);

        auto batchStart = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numThreads; i++)
        {
            threads.emplace_back([&]() {
                size_t qid;
                while ((qid = queriesSent.fetch_add(1)) < numQueries)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    index->SearchIndex(results[qid]);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    latencies[qid] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
                }
            });
        }

        for (auto &thread : threads)
            thread.join();

        auto batchEnd = std::chrono::high_resolution_clock::now();
        float batchLatency =
            std::chrono::duration_cast<std::chrono::microseconds>(batchEnd - batchStart).count() / 1000000.0f;

        float mean = 0;
        for (int i = 0; i < numQueries; i++)
        {
            mean += latencies[i];
        }
        mean /= numQueries;

        std::sort(latencies.begin(), latencies.end());
        float p95 = latencies[static_cast<size_t>(numQueries * 0.95)];
        float p99 = latencies[static_cast<size_t>(numQueries * 0.99)];
        float qps = numQueries / batchLatency;

        BOOST_TEST_MESSAGE("  Mean Latency: " << mean << " ms");
        BOOST_TEST_MESSAGE("  P95 Latency:  " << p95 << " ms");
        BOOST_TEST_MESSAGE("  P99 Latency:  " << p99 << " ms");
        BOOST_TEST_MESSAGE("  QPS:          " << qps);

        // Collect JSON data for Benchmark 5
        benchmark7Data << std::fixed << std::setprecision(4);
        benchmark7Data << "{\n";
        benchmark7Data << "      \"meanLatency\": " << mean << ",\n";
        benchmark7Data << "      \"p95\": " << p95 << ",\n";
        benchmark7Data << "      \"p99\": " << p99 << ",\n";
        benchmark7Data << "      \"qps\": " << qps << "\n";
        benchmark7Data << "    }";
    }

    // Benchmark 8: Recall evaluation (if truth file provided)
    BOOST_TEST_MESSAGE("Checking for truth file: " << truthPath);
    avgRecall = EvaluateRecall<T>(results, index, queryset, truth, vecset, addvecset, baseVectorCount, topK, 1);
    BOOST_TEST_MESSAGE("  Recall@" << topK << " = " << (avgRecall * 100.0f) << "%");
    BOOST_TEST_MESSAGE("  (Evaluated on " << numQueries << " queries against base vectors)");

    // Collect JSON data for Benchmark 6
    benchmark8Data << std::fixed << std::setprecision(4);
    benchmark8Data << "{\n";
    benchmark8Data << "      \"recallAtK\": " << avgRecall << ",\n";
    benchmark8Data << "      \"k\": " << topK << ",\n";
    benchmark8Data << "      \"numQueries\": " << numQueries << "\n";
    benchmark8Data << "    }";

    BOOST_TEST_MESSAGE("\n=== Benchmark Complete ===");
    // Write JSON output
    {
        std::ofstream jsonFile(outputFile);
        if (jsonFile.is_open())
        {
            jsonFile << std::fixed << std::setprecision(4);

            // Get current timestamp
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            std::tm tm_now;
            localtime_s(&tm_now, &time_t_now);

            std::ostringstream timestampStream;
            timestampStream << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%S");
            std::string timestamp = timestampStream.str();

            jsonFile << "{\n";
            jsonFile << "  \"timestamp\": \"" << timestamp << "\",\n";
            jsonFile << "  \"config\": {\n";
            jsonFile << "    \"vectorPath\": \"" << vectorPath << "\",\n";
            jsonFile << "    \"queryPath\": \"" << queryPath << "\",\n";
            jsonFile << "    \"truthPath\": \"" << truthPath << "\",\n";
            jsonFile << "    \"indexPath\": \"" << indexPath << "\",\n";
            jsonFile << "    \"dimension\": " << dimension << ",\n";
            jsonFile << "    \"baseVectorCount\": " << baseVectorCount << ",\n";
            jsonFile << "    \"insertVectorCount\": " << insertVectorCount << ",\n";
            jsonFile << "    \"topK\": " << topK << ",\n";
            jsonFile << "    \"numQueries\": " << numQueries << ",\n";
            jsonFile << "    \"numThreads\": " << numThreads << "\n";
            jsonFile << "  },\n";
            jsonFile << "  \"results\": {\n";
            jsonFile << "    \"benchmark1_insert\": " << benchmark1Data.str() << ",\n";
            jsonFile << "    \"benchmark2_query_before_save\": " << benchmark2Data.str() << ",\n";
            jsonFile << "    \"benchmark3_save\": " << benchmark3Data.str() << ",\n";
            jsonFile << "    \"benchmark4_reload\": " << benchmark4Data.str() << ",\n";
            jsonFile << "    \"benchmark5_query_after_reload\": " << benchmark5Data.str();
            if (!benchmark6Data.str().empty())
            {
                jsonFile << ",\n    \"benchmark6_recall\": " << benchmark6Data.str() << "\n";
            }
            else
            {
                jsonFile << "\n";
            }
            jsonFile << "    \"benchmark7_query_after_reload\": " << benchmark7Data.str();
            jsonFile << ",\n    \"benchmark8_recall\": " << benchmark8Data.str() << "\n";
            jsonFile << "  }\n";
            jsonFile << "}\n";

            jsonFile.close();
            BOOST_TEST_MESSAGE("\n=== Benchmark results saved to " << outputFile << " ===");
        }
        else
        {
            BOOST_TEST_MESSAGE("\n=== Failed to create " << outputFile << " ===");
        }
    }
}

} // namespace SPFreshTest

bool CompareFilesWithLogging(const std::filesystem::path &file1, const std::filesystem::path &file2)
{
    std::ifstream f1(file1, std::ios::binary);
    std::ifstream f2(file2, std::ios::binary);

    if (!f1.is_open() || !f2.is_open())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open one of the files:\n  %s\n  %s\n",
                     file1.string().c_str(), file2.string().c_str());
        return false;
    }

    // Check file sizes first
    f1.seekg(0, std::ios::end);
    f2.seekg(0, std::ios::end);
    if (f1.tellg() != f2.tellg())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File size differs: %s\n", file1.filename().string().c_str());
        return false;
    }

    f1.seekg(0, std::ios::beg);
    f2.seekg(0, std::ios::beg);

    const int bufferSize = 4096; // Adjust buffer size as needed
    std::vector<char> buffer1(bufferSize);
    std::vector<char> buffer2(bufferSize);

    while (f1.read(buffer1.data(), bufferSize) && f2.read(buffer2.data(), bufferSize))
    {
        if (std::memcmp(buffer1.data(), buffer2.data(), f1.gcount()) != 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File mismatch at: %s\n", file1.filename().string().c_str());
            return false; // Mismatch found
        }
    }

    return true;
}

bool CompareDirectoriesWithLogging(const std::filesystem::path &dir1, const std::filesystem::path &dir2,
                                   const std::unordered_set<std::string> &exceptions = {})
{
    std::map<std::string, std::filesystem::path> files1, files2;

    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir1))
    {
        if (entry.is_regular_file())
        {
            files1[std::filesystem::relative(entry.path(), dir1).string()] = entry.path();
        }
    }

    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir2))
    {
        if (entry.is_regular_file())
        {
            files2[std::filesystem::relative(entry.path(), dir2).string()] = entry.path();
        }
    }

    bool matched = true;

    for (const auto &[relPath, filePath1] : files1)
    {
        if (exceptions.count(relPath))
            continue;

        auto it = files2.find(relPath);
        if (it == files2.end())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Missing in %s: %s\n", dir2.string().c_str(), relPath.c_str());
            matched = false;
            continue;
        }
        if (!CompareFilesWithLogging(filePath1, it->second))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File end differs: %s\n", filePath1.filename().string().c_str());
            matched = false;
        }
    }

    for (const auto &[relPath, _] : files2)
    {
        if (exceptions.count(relPath))
            continue;
        if (files1.find(relPath) == files1.end())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Extra in %s: %s\n", dir2.string().c_str(), relPath.c_str());
            matched = false;
        }
    }

    return matched;
}

void NormalizeVector(float *embedding, int dimension)
{
    // get magnitude
    float magnitude = 0.0f;
    {
        float sum = 0.0;
        for (int i = 0; i < dimension; i++)
        {
            sum += embedding[i] * embedding[i];
        }
        magnitude = std::sqrt(sum);
    }

    // normalized target vector
    for (int i = 0; i < dimension; i++)
    {
        embedding[i] /= magnitude;
    }
}

template <typename T>
std::shared_ptr<VectorSet> get_embeddings(uint32_t row_id, uint32_t end_id, uint32_t embedding_dim,
                                          uint32_t array_index)
{
    uint32_t count = end_id - row_id;
    ByteArray vec = ByteArray::Alloc(sizeof(T) * count * embedding_dim);
    for (uint32_t rid = 0; rid < count; rid++)
    {
        for (int idx = 0; idx < embedding_dim; ++idx)
        {
            ((T *)vec.Data())[rid * embedding_dim + idx] = (row_id + rid) * 17 + idx * 19 + (array_index + 1) * 23;
        }
        NormalizeVector(((T *)vec.Data()) + rid * embedding_dim, embedding_dim);
    }
    return std::make_shared<BasicVectorSet>(vec, GetEnumValueType<T>(), embedding_dim, count);
}

BOOST_AUTO_TEST_SUITE(SPFreshTest)

BOOST_AUTO_TEST_CASE(TestLoadAndSave)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedIndex;
    BOOST_REQUIRE(VectorIndex::LoadIndex("original_index", loadedIndex) == ErrorCode::Success);
    BOOST_REQUIRE(loadedIndex != nullptr);
    BOOST_REQUIRE(loadedIndex->SaveIndex("loaded_and_saved_index") == ErrorCode::Success);
    loadedIndex = nullptr;

    std::unordered_set<std::string> exceptions = {"indexloader.ini"};

    // Compare files in both directories
    BOOST_REQUIRE_MESSAGE(CompareDirectoriesWithLogging("original_index", "loaded_and_saved_index", exceptions),
                          "Saved index does not match loaded-then-saved index");

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("loaded_and_saved_index");
}

BOOST_AUTO_TEST_CASE(TestReopenIndexRecall)
{
    using namespace SPFreshTest;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    float recall1 = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N);
    originalIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("original_index", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);
    BOOST_REQUIRE(loadedOnce->SaveIndex("reopened_index") == ErrorCode::Success);
    loadedOnce = nullptr;

    std::shared_ptr<VectorIndex> loadedTwice;
    BOOST_REQUIRE(VectorIndex::LoadIndex("reopened_index", loadedTwice) == ErrorCode::Success);
    BOOST_REQUIRE(loadedTwice != nullptr);
    float recall2 = Search<int8_t>(loadedTwice, queryset, vecset, addvecset, K, truth, N);
    loadedTwice = nullptr;

    BOOST_REQUIRE_MESSAGE(std::fabs(recall1 - recall2) < 1e-5, "Recall mismatch between original and reopened index");

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("reopened_index");
}

BOOST_AUTO_TEST_CASE(TestInsertAndSearch)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build base index
    auto index = BuildIndex<int8_t>("insert_test_index", vecset, metaset);
    BOOST_REQUIRE(index != nullptr);
    BOOST_REQUIRE(index->SaveIndex("insert_test_index") == ErrorCode::Success);
    index = nullptr;

    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_test_index", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);

    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loadedOnce.get()), 2, 1000, addvecset, addmetaset);
    SearchOnly<int8_t>(loadedOnce, queryset, K);
    loadedOnce = nullptr;

    std::filesystem::remove_all("insert_test_index");
}

BOOST_AUTO_TEST_CASE(TestClone)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);

    auto clonedIndex = originalIndex->Clone("cloned_index");
    BOOST_REQUIRE(clonedIndex != nullptr);
    BOOST_REQUIRE(clonedIndex->SaveIndex("cloned_index") == ErrorCode::Success);
    originalIndex.reset();
    clonedIndex = nullptr;

    std::unordered_set<std::string> exceptions = {"indexloader.ini"};

    // Compare files in both directories
    BOOST_REQUIRE_MESSAGE(CompareDirectoriesWithLogging("original_index", "cloned_index", exceptions),
                          "Saved index does not match loaded-then-saved index");

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("cloned_index");
}

BOOST_AUTO_TEST_CASE(TestCloneRecall)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    float originalRecall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N);

    auto clonedIndex = originalIndex->Clone("cloned_index");
    BOOST_REQUIRE(clonedIndex != nullptr);
    originalIndex.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClonedIndex;
    BOOST_REQUIRE(VectorIndex::LoadIndex("cloned_index", loadedClonedIndex) == ErrorCode::Success);
    BOOST_REQUIRE(loadedClonedIndex != nullptr);
    float clonedRecall = Search<int8_t>(loadedClonedIndex, queryset, vecset, addvecset, K, truth, N);
    loadedClonedIndex = nullptr;

    BOOST_REQUIRE_MESSAGE(std::fabs(originalRecall - clonedRecall) < 1e-5,
                          "Recall mismatch between original and cloned index: " << "original=" << originalRecall
                                                                                << ", cloned=" << clonedRecall);

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("cloned_index");
}

BOOST_AUTO_TEST_CASE(IndexPersistenceAndInsertSanity)
{
    using namespace SPFreshTest;

    // Prepare test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build and save base index
    auto baseIndex = BuildIndex<int8_t>("insert_test_index", vecset, metaset);
    BOOST_REQUIRE(baseIndex != nullptr);
    BOOST_REQUIRE(baseIndex->SaveIndex("insert_test_index") == ErrorCode::Success);
    baseIndex = nullptr;

    // Load the saved index
    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_test_index", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);

    // Search sanity check
    SearchOnly<int8_t>(loadedOnce, queryset, K);

    // Clone the loaded index
    auto clonedIndex = loadedOnce->Clone("insert_cloned_index");
    BOOST_REQUIRE(clonedIndex != nullptr);

    // Save and reload the cloned index
    BOOST_REQUIRE(clonedIndex->SaveIndex("insert_cloned_index") == ErrorCode::Success);
    loadedOnce.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClone;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_cloned_index", loadedClone) == ErrorCode::Success);
    BOOST_REQUIRE(loadedClone != nullptr);

    // Insert new vectors
    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loadedClone.get()), 1,
                          static_cast<int>(addvecset->Count()), addvecset, addmetaset);

    // Final save and reload after insert
    BOOST_REQUIRE(loadedClone->SaveIndex("insert_final_index") == ErrorCode::Success);
    loadedClone = nullptr;

    std::shared_ptr<VectorIndex> reloadedFinal;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_final_index", reloadedFinal) == ErrorCode::Success);

    // Final search sanity
    SearchOnly<int8_t>(reloadedFinal, queryset, K);
    reloadedFinal = nullptr;

    // Cleanup
    std::filesystem::remove_all("insert_test_index");
    std::filesystem::remove_all("insert_cloned_index");
    std::filesystem::remove_all("insert_final_index");
}

BOOST_AUTO_TEST_CASE(IndexPersistenceAndInsertMultipleThreads)
{
    using namespace SPFreshTest;

    // Prepare test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build and save base index
    auto baseIndex = BuildIndex<int8_t>("insert_test_index_multi", vecset, metaset);
    BOOST_REQUIRE(baseIndex != nullptr);
    BOOST_REQUIRE(baseIndex->SaveIndex("insert_test_index_multi") == ErrorCode::Success);
    baseIndex = nullptr;

    // Load the saved index
    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_test_index_multi", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);

    // Search sanity check
    SearchOnly<int8_t>(loadedOnce, queryset, K);

    // Clone the loaded index
    auto clonedIndex = loadedOnce->Clone("insert_cloned_index_multi");
    BOOST_REQUIRE(clonedIndex != nullptr);

    // Save and reload the cloned index
    BOOST_REQUIRE(clonedIndex->SaveIndex("insert_cloned_index_multi") == ErrorCode::Success);
    loadedOnce.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClone;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_cloned_index_multi", loadedClone) == ErrorCode::Success);
    BOOST_REQUIRE(loadedClone != nullptr);

    // Insert new vectors
    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loadedClone.get()), 2,
                          static_cast<int>(addvecset->Count()), addvecset, addmetaset);

    // Final save and reload after insert
    BOOST_REQUIRE(loadedClone->SaveIndex("insert_final_index_multi") == ErrorCode::Success);
    loadedClone = nullptr;

    std::shared_ptr<VectorIndex> reloadedFinal;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_final_index_multi", reloadedFinal) == ErrorCode::Success);
    BOOST_REQUIRE(reloadedFinal != nullptr);
    // Final search sanity
    SearchOnly<int8_t>(reloadedFinal, queryset, K);
    reloadedFinal = nullptr;

    // Cleanup
    std::filesystem::remove_all("insert_test_index_multi");
    std::filesystem::remove_all("insert_cloned_index_multi");
    std::filesystem::remove_all("insert_final_index_multi");
}

BOOST_AUTO_TEST_CASE(IndexSaveDuringQuery)
{
    using namespace SPFreshTest;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto index = BuildIndex<int8_t>("save_during_query_index", vecset, metaset);
    BOOST_REQUIRE(index != nullptr);

    std::atomic<bool> keepQuerying(true);
    std::thread queryThread([&]() {
        while (keepQuerying)
        {
            for (int q = 0; q < queryset->Count(); ++q)
            {
                QueryResult result(queryset->GetVector(q), K, true);
                index->SearchIndex(result);
            }
        }
    });

    // Wait a bit before saving
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ErrorCode saveStatus = index->SaveIndex("save_during_query_index");
    BOOST_REQUIRE(saveStatus == ErrorCode::Success);

    keepQuerying = false;
    queryThread.join();

    index = nullptr;

    std::shared_ptr<VectorIndex> reloaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex("save_during_query_index", reloaded) == ErrorCode::Success);
    BOOST_REQUIRE(reloaded != nullptr);

    SearchOnly<int8_t>(reloaded, queryset, K);
    reloaded = nullptr;

    std::filesystem::remove_all("save_during_query_index");
}

BOOST_AUTO_TEST_CASE(IndexMultiThreadedQuerySanity)
{
    using namespace SPFreshTest;

    // Generate test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build and save index
    auto index = BuildIndex<int8_t>("multi_query_index", vecset, metaset);
    BOOST_REQUIRE(index != nullptr);
    BOOST_REQUIRE(index->SaveIndex("multi_query_index") == ErrorCode::Success);
    index = nullptr;

    // Reload the index
    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex("multi_query_index", loaded) == ErrorCode::Success);
    BOOST_REQUIRE(loaded != nullptr);

    // Insert additional vectors
    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loaded.get()), 2, static_cast<int>(addvecset->Count()),
                          addvecset, addmetaset);

    // Perform multithreaded query
    const int threadCount = 4;
    std::vector<std::thread> threads;
    std::atomic<int> nextQuery(0);
    std::atomic<int> completedQueries(0);

    for (int t = 0; t < threadCount; ++t)
    {
        threads.emplace_back([&, t]() {
            QueryResult result(nullptr, K, true);
            while (true)
            {
                int i = nextQuery.fetch_add(1);
                if (i >= queryset->Count())
                    break;

                result.SetTarget(queryset->GetVector(static_cast<SizeType>(i)));
                loaded->SearchIndex(result);

                ++completedQueries;
            }
        });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Multithreaded query completed: %d queries\n", completedQueries.load());
    loaded = nullptr;

    // Cleanup
    std::filesystem::remove_all("multi_query_index");
}

BOOST_AUTO_TEST_CASE(IndexShadowCloneLifecycleKeepLast)
{
    using namespace SPFreshTest;

    constexpr int iterations = 5;
    constexpr int insertBatchSize = 100;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    const std::string baseIndexName = "base_index";
    BOOST_REQUIRE(BuildIndex<int8_t>(baseIndexName, vecset, metaset)->SaveIndex(baseIndexName) == ErrorCode::Success);

    std::string previousIndexName = baseIndexName;

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::string shadowIndexName = "shadow_index_" + std::to_string(iter);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[%d] Loading index: %s\n", iter, previousIndexName.c_str());

        // Load previous index
        std::shared_ptr<VectorIndex> loaded;
        BOOST_REQUIRE(VectorIndex::LoadIndex(previousIndexName, loaded) == ErrorCode::Success);
        BOOST_REQUIRE(loaded != nullptr);

        // Query check
        for (int i = 0; i < std::min<SizeType>(queryset->Count(), 5); ++i)
        {
            QueryResult result(queryset->GetVector(i), K, true);
            loaded->SearchIndex(result);
        }

        // Cleanup previous base index after first iteration
        if (iter == 1)
        {
            std::filesystem::remove_all(baseIndexName);
        }

        // Clone to shadow
        BOOST_REQUIRE(loaded->Clone(shadowIndexName) != nullptr);
        loaded.reset();

        std::shared_ptr<VectorIndex> shadowLoaded;
        BOOST_REQUIRE(VectorIndex::LoadIndex(shadowIndexName, shadowLoaded) == ErrorCode::Success);
        BOOST_REQUIRE(shadowLoaded != nullptr);
        auto *shadowIndex = static_cast<SPANN::Index<int8_t> *>(shadowLoaded.get());

        // Prepare insert batch
        const int insertOffset = (iter * insertBatchSize) % static_cast<int>(addvecset->Count());
        const int insertCount = min(insertBatchSize, static_cast<int>(addvecset->Count()) - insertOffset);

        std::vector<std::uint8_t> metaBytes;
        std::vector<std::uint64_t> offsetTable(insertCount + 1);
        std::uint64_t offset = 0;
        for (int i = 0; i < insertCount; ++i)
        {
            ByteArray meta = addmetaset->GetMetadata(insertOffset + i);
            offsetTable[i] = offset;
            metaBytes.insert(metaBytes.end(), meta.Data(), meta.Data() + meta.Length());
            offset += meta.Length();
        }
        offsetTable[insertCount] = offset;

        ByteArray metaBuf(new std::uint8_t[metaBytes.size()], metaBytes.size(), true);
        std::memcpy(metaBuf.Data(), metaBytes.data(), metaBytes.size());

        ByteArray offsetBuf(new std::uint8_t[offsetTable.size() * sizeof(std::uint64_t)],
                            offsetTable.size() * sizeof(std::uint64_t), true);
        std::memcpy(offsetBuf.Data(), offsetTable.data(), offsetTable.size() * sizeof(std::uint64_t));

        auto batchMeta = std::make_shared<MemMetadataSet>(metaBuf, offsetBuf, insertCount);
        const void *vectorStart = addvecset->GetVector(insertOffset);

        shadowIndex->AddIndex(vectorStart, insertCount, shadowIndex->GetOptions()->m_dim, batchMeta, true);

        while (!shadowIndex->AllFinished())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        BOOST_REQUIRE(shadowLoaded->SaveIndex(shadowIndexName) == ErrorCode::Success);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[%d] Created new shadow index: %s\n", iter, shadowIndexName.c_str());
        shadowLoaded = nullptr;

        previousIndexName = shadowIndexName;
    }

    // Keep the final shadow index directory for debugging/inspection
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Kept final index: %s\n", previousIndexName.c_str());

    // Cleanup all created indexes after test
    std::filesystem::remove_all(baseIndexName);
    for (int iter = 0; iter < iterations; ++iter)
    {
        std::string shadow = "shadow_index_" + std::to_string(iter);
        std::filesystem::remove_all(shadow);
    }
}

BOOST_AUTO_TEST_CASE(IterativeSearch)
{
    using namespace SPFreshTest;

    constexpr int insertIterations = 5;
    constexpr int insertBatchSize = 1000;
    constexpr int dimension = 1024;
    std::shared_ptr<VectorSet> vecset = get_embeddings<float>(0, insertBatchSize, dimension, -1);
    std::shared_ptr<MetadataSet> metaset = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, 0);

    auto originalIndex = BuildIndex<float>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::string prevPath = "original_index";
    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);

        auto cloneIndex = prevIndex->Clone(clone_path);
        auto *cloneIndexPtr = static_cast<SPANN::Index<float> *>(cloneIndex.get());
        std::shared_ptr<VectorSet> tmpvecs =
            get_embeddings<float>((iter + 1) * insertBatchSize, (iter + 2) * insertBatchSize, dimension, -1);
        std::shared_ptr<MetadataSet> tmpmetas =
            TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, (iter + 1) * insertBatchSize);
        InsertVectors<float>(cloneIndexPtr, 1, insertBatchSize, tmpvecs, tmpmetas);

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;

        std::shared_ptr<VectorIndex> loadedIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(clone_path, loadedIndex) == ErrorCode::Success);
        BOOST_REQUIRE(loadedIndex != nullptr);

        std::shared_ptr<VectorSet> embedding =
            get_embeddings<float>((1000 * iter) + 500, ((1000 * iter) + 501), dimension, -1);
        std::shared_ptr<ResultIterator> resultIterator = loadedIndex->GetIterator(embedding->GetData(), false);
        int batch = 100;
        int ri = 0;
        float current = INT_MAX, previous = INT_MAX;
        bool relaxMono = false;
        while (!relaxMono)
        {
            auto results = resultIterator->Next(batch);
            int resultCount = results->GetResultNum();
            if (resultCount <= 0)
                break;

            previous = current;
            current = 0;
            for (int j = 0; j < resultCount; j++)
            {
                std::cout << "Result[" << ri << "] VID:" << results->GetResult(j)->VID
                          << " Dist:" << results->GetResult(j)->Dist
                          << " RelaxedMono:" << results->GetResult(j)->RelaxedMono << " current:" << current
                          << " previous:" << previous << std::endl;
                relaxMono = results->GetResult(j)->RelaxedMono;
                current += results->GetResult(j)->Dist;
                ri++;
            }
            current /= resultCount;
        }
        resultIterator->Close();
        loadedIndex = nullptr;
    }

    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    std::filesystem::remove_all("original_index");
}

BOOST_AUTO_TEST_CASE(RefineIndex)
{
    using namespace SPFreshTest;

    int iterations = 5;
    int insertBatchSize = N / iterations;
    int deleteBatchSize = N / iterations;

    // Generate test data
    std::shared_ptr<VectorSet> vecset, addvecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, insertBatchSize, deleteBatchSize,
                         iterations, truth);

    // Build and save index
    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);

    float recall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N);
    std::cout << "original: recall@" << K << "= " << recall << std::endl;

    for (int iter = 0; iter < iterations; iter++)
    {
        InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(originalIndex.get()), 1, insertBatchSize, addvecset,
                              metaset, iter * insertBatchSize);
        for (int i = 0; i < deleteBatchSize; i++)
            originalIndex->DeleteIndex(iter * deleteBatchSize + i);

        recall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N, iter + 1);
        std::cout << "iter " << iter << ": recall@" << K << "=" << recall << std::endl;
    }
    std::cout << "Before Refine:" << " recall@" << K << "=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(originalIndex.get())->GetDBStat();
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    BOOST_REQUIRE(VectorIndex::LoadIndex("original_index", originalIndex) == ErrorCode::Success);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->Check() == ErrorCode::Success);

    recall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N, iterations);
    std::cout << "After Refine:" << " recall@" << K << "=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(originalIndex.get())->GetDBStat();
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::filesystem::remove_all("original_index");
}

BOOST_AUTO_TEST_CASE(CacheTest)
{
    using namespace SPFreshTest;

    int iterations = 5;
    int insertBatchSize = N / iterations;
    int deleteBatchSize = N / iterations;

    // Generate test data
    std::shared_ptr<VectorSet> vecset, addvecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, insertBatchSize, deleteBatchSize,
                         iterations, truth);

    // Build and save index
    std::shared_ptr<VectorIndex> originalIndex, finalIndex;

    std::filesystem::remove_all("original_index");

    originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    for (int iter = 0; iter < iterations; iter++)
    {
        if (direxists(("clone_index_" + std::to_string(iter)).c_str()))
        {
            std::filesystem::remove_all("clone_index_" + std::to_string(iter));
        }
    }

    std::string prevPath = "original_index";
    float recall = 0.0;

    std::cout << "=================No Cache===================" << std::endl;

    for (int iter = 0; iter < iterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);
        auto t0 = std::chrono::high_resolution_clock::now();
        BOOST_REQUIRE(prevIndex->Check() == ErrorCode::Success);
        std::cout << "[INFO] Check time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        auto cloneIndex = prevIndex->Clone(clone_path);
        prevIndex = nullptr;
        BOOST_REQUIRE(cloneIndex->Check() == ErrorCode::Success);

        recall = Search<int8_t>(cloneIndex, queryset, vecset, addvecset, K, truth, N, iter);
        std::cout << "[INFO] After Save, Clone and Load:" << " recall@" << K << "=" << recall << std::endl;
        static_cast<SPANN::Index<int8_t> *>(cloneIndex.get())->GetDBStat();

        auto t1 = std::chrono::high_resolution_clock::now();
        InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(cloneIndex.get()), 1, insertBatchSize, addvecset,
                              metaset, iter * insertBatchSize);
        std::cout << "[INFO] Insert time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t1)
                         .count()
                  << " ms" << std::endl;

        for (int i = 0; i < deleteBatchSize; i++)
            cloneIndex->DeleteIndex(iter * deleteBatchSize + i);

        recall = Search<int8_t>(cloneIndex, queryset, vecset, addvecset, K, truth, N, iter + 1);
        std::cout << "[INFO] After iter " << iter << ": recall@" << K << "=" << recall << std::endl;
        static_cast<SPANN::Index<int8_t> *>(cloneIndex.get())->GetDBStat();

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;
        prevPath = clone_path;
    }

    BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, finalIndex) == ErrorCode::Success);
    BOOST_REQUIRE(finalIndex != nullptr);
    auto t = std::chrono::high_resolution_clock::now();
    BOOST_REQUIRE(finalIndex->Check() == ErrorCode::Success);
    std::cout
        << "[INFO] Check time for iteration " << iterations << ": "
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t).count()
        << " ms" << std::endl;

    recall = Search<int8_t>(finalIndex, queryset, vecset, addvecset, K, truth, N, iterations);
    std::cout << "[INFO] After Save and Load:" << " recall@" << K << "=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(finalIndex.get())->GetDBStat();
    finalIndex = nullptr;
    for (int iter = 0; iter < iterations; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }

    std::cout << "=================Enable Cache===================" << std::endl;
    prevPath = "original_index";
    for (int iter = 0; iter < iterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);
        auto t0 = std::chrono::high_resolution_clock::now();
        BOOST_REQUIRE(prevIndex->Check() == ErrorCode::Success);
        std::cout << "[INFO] Check time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        prevIndex->SetParameter("CacheSizeGB", "4", "BuildSSDIndex");
        prevIndex->SetParameter("CacheShards", "2", "BuildSSDIndex");

        BOOST_REQUIRE(prevIndex->SaveIndex(prevPath) == ErrorCode::Success);
        auto cloneIndex = prevIndex->Clone(clone_path);

        recall = Search<int8_t>(cloneIndex, queryset, vecset, addvecset, K, truth, N, iter);
        std::cout << "[INFO] After Save, Clone and Load:" << " recall@" << K << "=" << recall << std::endl;
        static_cast<SPANN::Index<int8_t> *>(cloneIndex.get())->GetDBStat();

        auto t1 = std::chrono::high_resolution_clock::now();
        InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(cloneIndex.get()), 1, insertBatchSize, addvecset,
                              metaset, iter * insertBatchSize);
        std::cout << "[INFO] Insert time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t1)
                         .count()
                  << " ms" << std::endl;

        for (int i = 0; i < deleteBatchSize; i++)
            cloneIndex->DeleteIndex(iter * deleteBatchSize + i);

        recall = Search<int8_t>(cloneIndex, queryset, vecset, addvecset, K, truth, N, iter + 1);
        std::cout << "[INFO] After iter " << iter << ": recall@" << K << "=" << recall << std::endl;
        static_cast<SPANN::Index<int8_t> *>(cloneIndex.get())->GetDBStat();

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;
        prevPath = clone_path;
    }
    BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, finalIndex) == ErrorCode::Success);
    BOOST_REQUIRE(finalIndex != nullptr);
    auto tt = std::chrono::high_resolution_clock::now();
    BOOST_REQUIRE(finalIndex->Check() == ErrorCode::Success);
    std::cout
        << "[INFO] Check time for iteration " << iterations << ": "
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tt).count()
        << " ms" << std::endl;

    recall = Search<int8_t>(finalIndex, queryset, vecset, addvecset, K, truth, N, iterations);
    std::cout << "[INFO] After Save and Load:" << " recall@" << K << "=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(finalIndex.get())->GetDBStat();
    finalIndex = nullptr;

    for (int iter = 0; iter < iterations; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }

    std::filesystem::remove_all("original_index");
}

BOOST_AUTO_TEST_CASE(IterativeSearchPerf)
{
    using namespace SPFreshTest;

    constexpr int insertIterations = 5;
    constexpr int insertBatchSize = 60000;
    constexpr int appendBatchSize = 40000;
    constexpr int dimension = 100;
    std::shared_ptr<VectorSet> vecset = get_embeddings<float>(0, insertBatchSize, dimension, -1);
    std::shared_ptr<MetadataSet> metaset = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, 0);

    auto originalIndex = BuildIndex<float>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::string prevPath = "original_index";
    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);
        auto t0 = std::chrono::high_resolution_clock::now();
        BOOST_REQUIRE(prevIndex->Check() == ErrorCode::Success);
        std::cout << "Check time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        auto cloneIndex = prevIndex->Clone(clone_path);
        auto *cloneIndexPtr = static_cast<SPANN::Index<float> *>(cloneIndex.get());
        std::shared_ptr<VectorSet> tmpvecs = get_embeddings<float>(
            insertBatchSize + iter * appendBatchSize, insertBatchSize + (iter + 1) * appendBatchSize, dimension, -1);
        std::shared_ptr<MetadataSet> tmpmetas = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(
            appendBatchSize, insertBatchSize + (iter)*appendBatchSize);
        auto t1 = std::chrono::high_resolution_clock::now();
        InsertVectors<float>(cloneIndexPtr, 1, appendBatchSize, tmpvecs, tmpmetas);
        std::cout << "Insert time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t1)
                         .count()
                  << " ms" << std::endl;

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;
    }

    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    std::filesystem::remove_all("original_index");
}

BOOST_AUTO_TEST_CASE(RefineTestIdx)
{
    using namespace SPFreshTest;

    constexpr int dimension = 1024;

    std::shared_ptr<VectorSet> vecset = get_embeddings<float>(0, 500, dimension, -1);
    std::shared_ptr<MetadataSet> metaset = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(1000, 0);

    for (auto i = 0; i < 2; ++i)
    {
        void *p = vecset->GetVector(i);
        for (auto i = 0; i < dimension; ++i)
        {
            std::cout << ((float *)p)[i] << " ";
        }
        std::cout << std::endl;
    }

    auto originalIndex = BuildIndex<float>("original_index", vecset, metaset, "COSINE");
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::string prevPath = "original_index";
    for (int iter = 0; iter < 1; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);
        auto t0 = std::chrono::high_resolution_clock::now();
        BOOST_REQUIRE(prevIndex->Check() == ErrorCode::Success);
        std::cout << "Check time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        auto cloneIndex = prevIndex->Clone(clone_path);
        auto *cloneIndexPtr = static_cast<SPANN::Index<float> *>(cloneIndex.get());
        std::shared_ptr<VectorSet> tmpvecs = get_embeddings<float>(500, 1100, dimension, -1);
        std::shared_ptr<MetadataSet> tmpmetas = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(1200, 1000);
        auto t1 = std::chrono::high_resolution_clock::now();
        InsertVectors<float>(cloneIndexPtr, 1, 1200, tmpvecs, tmpmetas);
        std::cout << "Insert time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t1)
                         .count()
                  << " ms" << std::endl;

        for (auto i = 1000; i < 1900; ++i)
        {
            cloneIndexPtr->DeleteIndex(i);
        }

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;
    }

    for (int iter = 0; iter < 1; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    // std::filesystem::remove_all("original_index");
}

BOOST_AUTO_TEST_CASE(BenchmarkFromConfig)
{
    using namespace SPFreshTest;

    // Check if benchmark config is provided via environment variable
    const char *configPath = std::getenv("BENCHMARK_CONFIG");
    if (configPath == nullptr)
    {
        BOOST_TEST_MESSAGE("Skipping benchmark test - BENCHMARK_CONFIG environment variable not set");
        return;
    }

    BOOST_TEST_MESSAGE("Running benchmark with config: " << configPath);

    // Read benchmark configuration
    Helper::IniReader iniReader;
    if (ErrorCode::Success != iniReader.LoadIniFile(configPath))
    {
        BOOST_FAIL("Failed to load benchmark config file: " << configPath);
        return;
    }

    // Parse config parameters
    std::string vectorPath = iniReader.GetParameter("Benchmark", "VectorPath", std::string(""));
    std::string queryPath = iniReader.GetParameter("Benchmark", "QueryPath", std::string(""));
    std::string truthPath = iniReader.GetParameter("Benchmark", "TruthPath", std::string(""));
    std::string indexPath = iniReader.GetParameter("Benchmark", "IndexPath", std::string("benchmark_index"));

    VectorValueType valueType = VectorValueType::Float;
    std::string valueTypeStr = iniReader.GetParameter("Benchmark", "ValueType", std::string("Float"));
    if (valueTypeStr == "Float")
        valueType = VectorValueType::Float;
    else if (valueTypeStr == "Int8")
        valueType = VectorValueType::Int8;
    else if (valueTypeStr == "UInt8")
        valueType = VectorValueType::UInt8;

    int dimension = iniReader.GetParameter("Benchmark", "Dimension", 128);
    int baseVectorCount = iniReader.GetParameter("Benchmark", "BaseVectorCount", 10000);
    int insertVectorCount = iniReader.GetParameter("Benchmark", "InsertVectorCount", 2000);
    int batchSize = iniReader.GetParameter("Benchmark", "BatchSize", 100);
    int topK = iniReader.GetParameter("Benchmark", "TopK", 10);
    int numThreads = iniReader.GetParameter("Benchmark", "NumThreads", 32);
    int numQueries = iniReader.GetParameter("Benchmark", "NumQueries", 1000);

    // Parse SelectHead/Build configuration parameters
    BuildConfig buildConfig;
    buildConfig.selectHeadThreads = iniReader.GetParameter("SelectHead", "NumberOfThreads", 16);
    buildConfig.buildHeadThreads = iniReader.GetParameter("BuildHead", "NumberOfThreads", 16);
    buildConfig.buildSSDThreads = iniReader.GetParameter("BuildSSDIndex", "NumberOfThreads", 16);
    buildConfig.parallelBKTBuild = iniReader.GetParameter("SelectHead", "ParallelBKTBuild", false);
    buildConfig.bktKmeansK = iniReader.GetParameter("SelectHead", "BKTKmeansK", 32);
    buildConfig.bktLeafSize = iniReader.GetParameter("SelectHead", "BKTLeafSize", 8);
    buildConfig.bktLambdaFactor = iniReader.GetParameter("SelectHead", "BKTLambdaFactor", -1.0f);
    buildConfig.ratio = iniReader.GetParameter("SelectHead", "Ratio", 0.2);
    buildConfig.selectThreshold = iniReader.GetParameter("SelectHead", "SelectThreshold", 0);
    buildConfig.splitFactor = iniReader.GetParameter("SelectHead", "SplitFactor", 0);
    buildConfig.splitThreshold = iniReader.GetParameter("SelectHead", "SplitThreshold", 0);

    BOOST_TEST_MESSAGE("=== Benchmark Configuration ===");
    BOOST_TEST_MESSAGE("Vector Path: " << vectorPath);
    BOOST_TEST_MESSAGE("Query Path: " << queryPath);
    BOOST_TEST_MESSAGE("Base Vectors: " << baseVectorCount);
    BOOST_TEST_MESSAGE("Insert Vectors: " << insertVectorCount);
    BOOST_TEST_MESSAGE("Dimension: " << dimension);
    BOOST_TEST_MESSAGE("Batch Size: " << batchSize);
    BOOST_TEST_MESSAGE("Top-K: " << topK);
    BOOST_TEST_MESSAGE("Threads: " << numThreads);
    BOOST_TEST_MESSAGE("Queries: " << numQueries);
    BOOST_TEST_MESSAGE("Parallel BKT Build: " << (buildConfig.parallelBKTBuild ? "true" : "false"));
    BOOST_TEST_MESSAGE("SelectHead Threads: " << buildConfig.selectHeadThreads);
    BOOST_TEST_MESSAGE("BKTKmeansK: " << buildConfig.bktKmeansK);
    BOOST_TEST_MESSAGE("BKTLeafSize: " << buildConfig.bktLeafSize);

    // Get output file path from environment variable or use default
    const char *outputPath = std::getenv("BENCHMARK_OUTPUT");
    std::string outputFile = outputPath ? std::string(outputPath) : "output.json";
    BOOST_TEST_MESSAGE("Output File: " << outputFile);

    // Dispatch to appropriate type
    if (valueType == VectorValueType::Float)
    {
        RunBenchmark<float>(vectorPath, queryPath, truthPath, indexPath, dimension, baseVectorCount, insertVectorCount,
                            batchSize, topK, numThreads, numQueries, outputFile, buildConfig);
    }
    else if (valueType == VectorValueType::Int8)
    {
        RunBenchmark<std::int8_t>(vectorPath, queryPath, truthPath, indexPath, dimension, baseVectorCount,
                                  insertVectorCount, batchSize, topK, numThreads, numQueries, outputFile, buildConfig);
    }
    else if (valueType == VectorValueType::UInt8)
    {
        RunBenchmark<std::uint8_t>(vectorPath, queryPath, truthPath, indexPath, dimension, baseVectorCount,
                                   insertVectorCount, batchSize, topK, numThreads, numQueries, outputFile, buildConfig);
    }
}

BOOST_AUTO_TEST_SUITE_END()
