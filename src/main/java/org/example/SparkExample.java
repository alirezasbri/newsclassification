package org.example;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;

public class SparkExample {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("SimpleApp").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        JavaRDD<String> data = sc.parallelize(java.util.Arrays.asList("Hello", "World", "from", "Spark"));
        long count = data.count();

        System.out.println("Total words: " + count);

        sc.close();
    }
}