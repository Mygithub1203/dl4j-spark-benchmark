package org.deeplearning4j.train.config;

import java.util.Random;

/**
 * Created by Alex on 23/07/2016.
 */
public abstract class BaseSparkTest implements SparkTest {

    protected int minibatchSizePerWorker;
    protected int dataSize;
    protected int paramsSize;
    protected Random rng = new Random();


    protected BaseSparkTest(Builder builder){
        this.minibatchSizePerWorker = builder.minibatchSizePerWorker;
        this.dataSize = builder.dataSize;
        this.paramsSize = builder.paramsSize;
    }


    @SuppressWarnings("unchecked")
    public static abstract class Builder<T extends Builder<T>>{

        protected int minibatchSizePerWorker = 32;
        protected int dataSize;
        protected int paramsSize;
        protected Random rng = new Random();

        public T minibatchSizePerWorker(int minibatchSizePerWorker){
            this.minibatchSizePerWorker = minibatchSizePerWorker;
            return (T)this;
        }

        public T paramsSize(int paramsSize){
            this.paramsSize = paramsSize;
            return (T)this;
        }

        public T dataSize(int dataSize){
            this.dataSize = dataSize;
            return (T)this;
        }


    }

}
