package org.deeplearning4j.train.config;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.train.enums.DataSize;
import org.deeplearning4j.train.enums.ParamsSize;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;

/**
 * Created by Alex on 23/07/2016.
 */
public interface SparkTest extends Serializable {

    MultiLayerConfiguration getConfiguration();

    DataSet getSyntheticDataSet();

}
