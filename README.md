# dl4j-spark-benchmark
DL4J Benchmarking code for Spark

Tests are based on synthetic data, and are fully parameterized, with respect to network and
data sizes, etc.

Can be run either on Spark local or via spark-submit.

For building uber-jar for Spark local:
mvn package -DskipTests -P sparklocal

For building uber-jar for Spark submit:
mvn package -DskipTests

Example launch shell scripts are available under /scripts