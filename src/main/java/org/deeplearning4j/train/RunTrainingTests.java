package org.deeplearning4j.train;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.deeplearning4j.train.config.MLPTest;
import org.deeplearning4j.train.config.SparkTest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Alex on 23/07/2016.
 */
public class RunTrainingTests {

    @Parameter(names="-numTestFiles", description = "Number of test files (DataSet objects)")
    protected int numTestFiles = 10000;

    @Parameter(names="-tempPath", description = "Path to the test directory (typically HDFS), in which to generate data", required = true)
    protected String tempPath;

    @Parameter(names="-resultPath", description = "Path to the base output directory. Results will be placed in a subdirectory. For example, HDFS or S3", required = true)
    protected String resultPath;

    @Parameter(names="-numParams", variableArity = true, description = "Number of parameters in the network, as a list: \"-numParams 100000 1000000 10000000\"")
    protected List<Integer> numParams = new ArrayList<>(Arrays.asList(100_000,1_000_000,10_000_000));

    @Parameter(names="-dataSize", variableArity = true, description = "Size of the data set (i.e., num inputs/outputs)")
    protected List<Integer> dataSize = new ArrayList<>(Arrays.asList(16,128,512,2048));

    @Parameter(names="-miniBatchSizePerWorker", variableArity = true, description = "Number of examples per worker/minibatch, as a list: \"-miniBatchSizePerWorker 8 32 128\"")
    protected List<Integer> miniBatchSizePerWorker = new ArrayList<>(Arrays.asList(8, 32, 128));

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


        List<SparkTest> testsToRun = new ArrayList<>();
        for(Integer np : numParams){
            for(Integer ds : dataSize){
                for(Integer mbs : miniBatchSizePerWorker){

                    testsToRun.add(
                            new MLPTest.Builder()
                                    .paramsSize(np)
                                    .dataSize(ds)
                                    .minibatchSizePerWorker(mbs).build());

                }
            }
        }



        int test = 0;
        for(SparkTest st : testsToRun){





            test++;
        }



    }

}
