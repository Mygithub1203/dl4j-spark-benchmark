package org.deeplearning4j.train;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.data.DataSetExportFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.train.config.MLPTest;
import org.deeplearning4j.train.config.SparkTest;
import org.deeplearning4j.train.functions.GenerateDataFunction;
import org.nd4j.linalg.dataset.DataSet;

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Alex on 23/07/2016.
 */
@Slf4j
public class RunTrainingTests {



    @Parameter(names="-numTestFiles", description = "Number of test files (DataSet objects)")
    protected int numTestFiles = 10000;

    @Parameter(names="-tempPath", description = "Path to the test directory (typically HDFS), in which to generate data", required = true)
    protected String tempPath;

    @Parameter(names="-resultPath", description = "Path to the base output directory. Results will be placed in a subdirectory. For example, HDFS or S3", required = true)
    protected String resultPath;

    @Parameter(names="-skipExisting", description = "Flag to skip (don't re-run) any tests that have already been completed")
    protected boolean skipExisting;

    @Parameter(names="-numParams", variableArity = true, description = "Number of parameters in the network, as a list: \"-numParams 100000 1000000 10000000\"")
    protected List<Integer> numParams = new ArrayList<>(Arrays.asList(100_000,1_000_000,10_000_000));

    @Parameter(names="-dataSize", variableArity = true, description = "Size of the data set (i.e., num inputs/outputs)")
    protected List<Integer> dataSize = new ArrayList<>(Arrays.asList(16,128,512,2048));

    @Parameter(names="-miniBatchSizePerWorker", variableArity = true, description = "Number of examples per worker/minibatch, as a list: \"-miniBatchSizePerWorker 8 32 128\"")
    protected List<Integer> miniBatchSizePerWorker = new ArrayList<>(Arrays.asList(8, 32, 128));

    @Parameter(names="-saveUpdater", description = "Whether the updater should be saved or not", arity = 1)
    protected boolean saveUpdater = true;

    @Parameter(names="-repartition", description = "When repartitioning should occur")
    protected Repartition repartition = Repartition.Always;

    @Parameter(names="-repartitionStrategy", description = "Repartition strategy to use when repartitioning")
    protected RepartitionStrategy repartitionStrategy = RepartitionStrategy.SparkDefault;

    @Parameter(names="-workerPrefetchNumBatches", description = "Number of batches to prefetch")
    protected int workerPrefetchNumBatches = 0;


    public static void main(String[] args) throws Exception {
        new RunTrainingTests().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        SparkConf conf = new SparkConf();
        conf.setAppName("RunTrainingTests");
        JavaSparkContext sc = new JavaSparkContext(conf);

        List<SparkTest> testsToRun = new ArrayList<>();
        for(Integer np : numParams){
            for(Integer ds : dataSize){
                for(Integer mbs : miniBatchSizePerWorker){

                    testsToRun.add(
                            new MLPTest.Builder()
                                    .paramsSize(np)
                                    .dataSize(ds)
                                    .minibatchSizePerWorker(mbs)
                                    .saveUpdater(saveUpdater)
                                    .repartition(repartition)
                                    .repartitionStrategy(repartitionStrategy)
                                    .workerPrefetchNumBatches(workerPrefetchNumBatches)
                                    .build());

                }
            }
        }

        List<Integer> intList = new ArrayList<>();
        JavaRDD<Integer> intRDD = sc.parallelize(intList);

        Configuration config = new Configuration();
        FileSystem hdfs = FileSystem.get(URI.create(tempPath), config);

        int test = 0;
        for(SparkTest st : testsToRun){

            String dataDir = tempPath + (tempPath.endsWith("/") ? "" : "/") + test + "/";

            boolean exists = hdfs.exists(new Path(tempPath));
            if(exists){
                log.info("Temporary directory exists; attempting to delete. {}", tempPath);
                hdfs.delete(new Path(tempPath), true);
            }



            //Step 1: generate data
            long startGenerateExport = System.currentTimeMillis();
            JavaRDD<DataSet> data = intRDD.map(new GenerateDataFunction(st));
            data.foreachPartition(new DataSetExportFunction(new URI(dataDir)));
            long endGenerateExport = System.currentTimeMillis();


            //Step 2: Train network for 1 epoch
            MultiLayerConfiguration netConf = st.getConfiguration();

            TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(st.getDataSize())
                    .saveUpdater(st.isSaveUpdater())
                    .repartionData(st.getRepartition())
                    .repartitionStrategy(st.getRepartitionStrategy())
                    .workerPrefetchNumBatches(st.getWorkerPrefetchNumBatches())
                    .build();

            SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc, netConf, tm);


            //Step 3: record results




            test++;
        }



    }

}
