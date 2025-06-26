package org.example;


import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.WriteConfig;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.*;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.*;

import java.util.HashMap;
import java.util.Map;

public class NewsClassifierWithAI {

    public static void main(String[] args) {
        // Create Spark session
        SparkSession spark = SparkSession.builder()
                .appName("MongoDB News Classifier")
                .master("local[*]")
                .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/newsdb2.news")
                .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/newsdb2.classifiednews")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        // Load data from MongoDB
        Dataset<Row> data = spark.read()
                .format("mongo")
                .option("uri", "mongodb://127.0.0.1/newsdb2.news")
                .load();

        // Select relevant fields
        Dataset<Row> cleaned = data.select("headline", "short_description", "category")  // 'label' must exist in DB
                .na().drop(); // drop rows with nulls

        // Text processing
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("short_description")
                .setOutputCol("words");

        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filtered");

        HashingTF tf = new HashingTF()
                .setInputCol("filtered")
                .setOutputCol("rawFeatures")
                .setNumFeatures(1000);

        IDF idf = new IDF()
                .setInputCol("rawFeatures")
                .setOutputCol("features");

        // Logistic regression model
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setLabelCol("category");

        // Build pipeline
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                tokenizer, remover, tf, idf, lr
        });

        // Split data into training and testing
        Dataset<Row>[] splits = cleaned.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train model
        PipelineModel model = pipeline.fit(trainingData);

        // Predict
        Dataset<Row> predictions = model.transform(testData);
        predictions.select("headline", "short_description", "category", "prediction").show(false);

        // Prepare write config
        Map<String, String> writeOverrides = new HashMap<>();
        writeOverrides.put("collection", "classifiednews");
        writeOverrides.put("writeConcern.w", "majority");

        WriteConfig writeConfig = WriteConfig.create(spark).withOptions(writeOverrides);

        // Save back to MongoDB
        MongoSpark.write(predictions.select("headline", "short_description", "category", "prediction"))
                .option("collection", "classifiednews")
                .mode("overwrite")
                .save();

        spark.stop();
    }
}

