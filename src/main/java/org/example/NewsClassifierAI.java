package org.example;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;

import java.util.Arrays;

public class NewsClassifierAI {
    public static void main(String[] args) {
        // Spark configuration
        SparkConf conf = new SparkConf()
                .setAppName("NewsClassifierAI")
                .setMaster("local[*]")
                .set("spark.mongodb.input.uri", "mongodb://127.0.0.1/newsdb2.news")
                .set("spark.mongodb.output.uri", "mongodb://127.0.0.1/newsdb2.classifiednews");

        SparkSession spark = SparkSession.builder()
                .config(conf)
                .getOrCreate();

        // Read from MongoDB
        Dataset<Row> df = spark.read()
                .format("com.mongodb.spark.sql.DefaultSource")
                .load();

        df.printSchema();

        // Check correct text field
        if (!Arrays.asList(df.columns()).contains("description")) {
            System.err.println("Column 'description' not found!");
            spark.stop();
            return;
        }

        // Filter nulls
        Dataset<Row> filtered = df.filter("description IS NOT NULL AND category IS NOT NULL");

        // String indexer for label
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("category")
                .setOutputCol("label")
                .setHandleInvalid("skip");

        // Tokenize text
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("description")
                .setOutputCol("words");

        // TF
        HashingTF tf = new HashingTF()
                .setInputCol("words")
                .setOutputCol("features")
                .setNumFeatures(16384);

        // Classifier
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.001);

        // Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{labelIndexer, tokenizer, tf, lr});

        // Fit model
        PipelineModel model = pipeline.fit(filtered);

        // Predict
        Dataset<Row> predictions = model.transform(filtered);

        // Save selected output (no features column!)
        Dataset<Row> toSave = predictions.select("category", "description", "prediction");

        toSave.write()
                .format("com.mongodb.spark.sql.DefaultSource")
                .mode("overwrite")
                .save();

        spark.stop();
    }
}
