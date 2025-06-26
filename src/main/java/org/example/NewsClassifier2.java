package org.example;

import com.mongodb.spark.config.ReadConfig;
import com.mongodb.spark.config.WriteConfig;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.*;
import org.apache.spark.sql.functions;

import java.util.HashMap;
import java.util.Map;

public class NewsClassifier2 {

    public static void main(String[] args) {
        // Spark Configuration
        SparkConf conf = new SparkConf()
                .setAppName("News Classification")
                .setMaster("local[*]")
                .set("spark.mongodb.input.uri", "mongodb://127.0.0.1/newsdb.news")
                .set("spark.mongodb.output.uri", "mongodb://127.0.0.1/newsdb.classifiednews");

        JavaSparkContext jsc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(jsc);

        // Read from MongoDB
        Map<String, String> readOverrides = new HashMap<>();
        readOverrides.put("uri", "mongodb://127.0.0.1/newsdb.news");

        Dataset<Row> df = com.mongodb.spark.MongoSpark.load(jsc, ReadConfig.create(jsc)).toDF();

        // classification
        Dataset<Row> classified = df.withColumn("category", functions.when(
                        functions.col("title").contains("business")
                                .or(functions.col("description").contains("business")), "Business")
                .when(functions.col("title").contains("sports")
                        .or(functions.col("description").contains("sports")), "Sports")
                .when(functions.col("title").contains("tech")
                        .or(functions.col("description").contains("tech"))
                        .or(functions.col("title").contains("AI"))
                        .or(functions.col("description").contains("AI")), "Technology")
                .when(functions.col("title").contains("election")
                        .or(functions.col("description").contains("government"))
                        .or(functions.col("description").contains("president")), "Politics")
                .when(functions.col("title").contains("health")
                        .or(functions.col("description").contains("health"))
                        .or(functions.col("description").contains("vaccine")), "Health")
                .otherwise("Other")
        );


//        classified.show(10, false);

        // Write back to MongoDB
        Map<String, String> writeOverrides = new HashMap<>();
        writeOverrides.put("uri", "mongodb://127.0.0.1/newsdb.classifiednews");

        com.mongodb.spark.MongoSpark.save(classified.write().mode("overwrite"), WriteConfig.create(jsc));

        jsc.close();
    }
}

