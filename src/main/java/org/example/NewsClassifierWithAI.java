package org.example;

import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import com.mongodb.spark.config.WriteConfig;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.*;

import java.util.HashMap;
import java.util.Map;

public class NewsClassifierWithAI {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setAppName("NewsClassifierWithAI")
                .setMaster("local")
                .set("spark.mongodb.input.uri", "mongodb://127.0.0.1/newsdb2.news")
                .set("spark.mongodb.output.uri", "mongodb://127.0.0.1/newsdb2.classifiednews");

        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        // Read from MongoDB
        Map<String, String> readOverrides = new HashMap<>();
        readOverrides.put("uri", "mongodb://127.0.0.1");
        readOverrides.put("database", "newsdb2");
        readOverrides.put("collection", "news");
        ReadConfig readConfig = ReadConfig.create(spark).withOptions(readOverrides);
        Dataset<Row> df = spark.read()
                .format("com.mongodb.spark.sql.DefaultSource")
                .options(readOverrides)
                .load();

        // For simplicity, we'll use the 'description' as input text and assume a 'category' field exists
        df = df.filter("description IS NOT NULL AND category IS NOT NULL");

        // String label to numeric
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("category")
                .setOutputCol("label")
                .setHandleInvalid("skip");

        // Text processing pipeline
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("description")
                .setOutputCol("words");

        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filtered");

        HashingTF hashingTF = new HashingTF()
                .setInputCol("filtered")
                .setOutputCol("features")
                .setNumFeatures(1000);

        // Model
        NaiveBayes nb = new NaiveBayes();

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                labelIndexer, tokenizer, remover, hashingTF, nb
        });

        // Split data
        Dataset<Row>[] splits = df.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train and predict
        PipelineModel model = pipeline.fit(trainingData);
        Dataset<Row> predictions = model.transform(testData);

        // Evaluate
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy = " + accuracy);

//        DataFrameToSave.drop("features").write()
//                .format("mongo")
//                .mode("overwrite")
//                .option("uri", "mongodb://localhost:27017/news_db.predictions")
//                .save();

        // Save to MongoDB
        Map<String, String> writeOverrides = new HashMap<>();
        writeOverrides.put("uri", "mongodb://127.0.0.1");
        writeOverrides.put("database", "newsdb2");
        writeOverrides.put("collection", "classifiednews");
        WriteConfig writeConfig = WriteConfig.create(spark).withOptions(writeOverrides);

        MongoSpark.write(predictions).option("collection", "classifiednews").mode("overwrite").save();

        spark.stop();
    }
}
