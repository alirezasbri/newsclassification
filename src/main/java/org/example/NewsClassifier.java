package org.example;

import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.bson.Document;

import java.util.HashMap;
import java.util.Map;

public class NewsClassifier {

    public static void main(String[] args) {
        // Spark configuration
        SparkConf conf = new SparkConf()
                .setAppName("NewsClassifier")
                .setMaster("local[*]")
                .set("spark.mongodb.input.uri", "mongodb://127.0.0.1/newsdb.news")
                .set("spark.mongodb.output.uri", "mongodb://127.0.0.1/newsdb.classifiednews");

        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");

        // Load from MongoDB
        Map<String, String> readOverrides = new HashMap<>();
        readOverrides.put("uri", "mongodb://127.0.0.1/newsdb.news");
        ReadConfig readConfig = ReadConfig.create(sc).withOptions(readOverrides);
        JavaRDD<Document> rdd = MongoSpark.load(sc, readConfig);

        // Classify news by keywords in title/description
        JavaRDD<Document> classified = rdd.map(doc -> {
            String title = doc.getString("title");
            String description = doc.getString("description");
            String category;

            if (title.toLowerCase().contains("economy") || description.toLowerCase().contains("market")) {
                category = "Business";
            } else if (title.toLowerCase().contains("sports") || description.toLowerCase().contains("match")) {
                category = "Sports";
            } else {
                category = "General";
            }

            doc.append("category", category);
            return doc;
        });

        // Save back to MongoDB
        MongoSpark.save(classified);

        sc.close();
    }
}
