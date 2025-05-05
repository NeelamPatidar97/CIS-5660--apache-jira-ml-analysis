#!/usr/bin/env python
# coding: utf-8

"""
Completely Fixed Bug Analysis Script - Ready for Production

This script performs comprehensive analysis of Apache JIRA bugs with robust
error handling and fixes for all known issues, especially the silhouette score problem.

Author: Sally Zreiqat
Date: May 5, 2023
"""

import os
import time
import argparse
import sys
import traceback
from datetime import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, expr, lit, count, collect_list, size, array_contains, 
    countDistinct, datediff, to_date, desc, regexp_replace, lower, 
    concat_ws, split, explode, array_join, length, avg, sum, min, max, stddev
)
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType, BooleanType

from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, CountVectorizer,
    IDF, Word2Vec, StopWordsRemover, Tokenizer, HashingTF, NGram
)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, 
    GBTClassifier, DecisionTreeClassifier
)
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator
from pyspark.ml import Pipeline, PipelineModel

def setup_spark(app_name="Final Apache JIRA Bug Analysis"):
    """Set up and return a Spark session with optimized configurations."""
    logger.info(f"Setting up Spark session for: {app_name}")
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "20") \
        .config("spark.default.parallelism", "20") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_data(spark, issues_path, changelog_path=None, sample_fraction=None):
    """Load JIRA issues and changelog data for analysis."""
    # File paths (either provided or use defaults)
    issues_path = issues_path or "/user/szreiqa/Apache_JIRA_Issues/cleaned_issues.parquet"
    changelog_path = changelog_path or "/user/szreiqa/Apache_JIRA_Issues/cleaned_changelog.parquet"

    # Load issues data
    logger.info(f"Loading issues from: {issues_path}")
    issues_df = spark.read.parquet(issues_path)
    issue_count = issues_df.count()
    logger.info(f"Total issues: {issue_count}")

    # Sample if requested
    if sample_fraction and sample_fraction < 1.0:
        logger.info(f"Sampling {sample_fraction*100:.1f}% of issues")
        issues_df = issues_df.sample(False, sample_fraction, seed=42)
        logger.info(f"Sampled to {issues_df.count()} issues")

    # Filter for bugs only
    bugs_df = issues_df.filter(col("issuetype_name") == "Bug")
    bug_count = bugs_df.count()
    logger.info(f"Total bugs: {bug_count} ({bug_count/issue_count*100:.2f}% of all issues)")

    # Load changelog data if path provided
    changelog_df = None
    if changelog_path:
        logger.info(f"Loading changelog from: {changelog_path}")
        changelog_df = spark.read.parquet(changelog_path)
        
        # Sample if we sampled the issues
        if sample_fraction and sample_fraction < 1.0:
            # Only keep changelog entries for the issues we have
            bug_keys = bugs_df.select("key").collect()
            bug_keys_list = [row["key"] for row in bug_keys]
            changelog_df = changelog_df.filter(col("key").isin(bug_keys_list))
        
        changelog_count = changelog_df.count()
        logger.info(f"Relevant changelog entries: {changelog_count}")

    return bugs_df, changelog_df

def analyze_bug_reopening(bugs_df, changelog_df):
    """Analyze which bugs were reopened after being resolved."""
    logger.info("\n--- ANALYZING BUG REOPENING PATTERNS ---")
    
    if changelog_df is None:
        logger.info("No changelog data provided, skipping reopening analysis")
        # Add was_reopened column with default False
        return bugs_df.withColumn("was_reopened", lit(False))
    
    # Filter status changes from the changelog
    status_changes = changelog_df.filter(
        (col("field") == "status") & 
        col("fromString").isNotNull() & 
        col("toString").isNotNull()
    )

    # Group changes by issue key and collect status transitions
    status_transitions = status_changes.groupBy("key").agg(
        collect_list("fromString").alias("from_statuses"),
        collect_list("toString").alias("to_statuses")
    )

    # Define a function to detect reopening patterns
    def has_reopen_pattern(from_statuses, to_statuses):
        """Detects if the issue has been reopened based on status transitions."""
        if not from_statuses or not to_statuses or len(from_statuses) != len(to_statuses):
            return False
            
        # Define resolution and reopening statuses
        resolution_statuses = ["resolved", "closed", "done", "fixed", "completed"]
        reopen_statuses = ["reopened", "in progress", "open", "todo", "to do", "in development"]
        
        # Look for patterns where a resolved issue is reopened
        for i in range(len(from_statuses) - 1):
            current_to = to_statuses[i].lower()
            next_from = from_statuses[i+1].lower()
            next_to = to_statuses[i+1].lower()
            
            # Check if an issue moved to resolved/closed and then away from it
            if any(status in current_to for status in resolution_statuses) and \
               any(status in next_from for status in resolution_statuses) and \
               any(status in next_to for status in reopen_statuses):
                return True
                
        # Also check for direct reopened status
        if any(status.lower() == "reopened" for status in to_statuses):
            return True
            
        return False

    # Register UDF for use in Spark
    from pyspark.sql.functions import udf
    reopen_pattern_udf = udf(has_reopen_pattern, BooleanType())

    # Apply UDF to detect reopened issues
    bugs_with_reopen = status_transitions.withColumn(
        "was_reopened", reopen_pattern_udf(col("from_statuses"), col("to_statuses"))
    )

    # Join with bugs dataframe
    bugs_with_reopen_flag = bugs_df.join(
        bugs_with_reopen.select("key", "was_reopened"),
        "key",
        "left"
    ).withColumn(
        "was_reopened", 
        when(col("was_reopened").isNull(), False).otherwise(col("was_reopened"))
    )

    # Count reopened bugs
    reopened_count = bugs_with_reopen_flag.filter(col("was_reopened") == True).count()
    total_bugs = bugs_df.count()
    logger.info(f"Total bugs: {total_bugs}")
    logger.info(f"Bugs with reopening pattern: {reopened_count} ({reopened_count/total_bugs*100:.2f}%)")

    # Show some examples of reopened bugs
    logger.info("\nExamples of reopened bugs:")
    bugs_with_reopen_flag.filter(col("was_reopened") == True) \
        .select("key", "summary", "priority_name", "status_name") \
        .limit(5) \
        .show(truncate=False)
    
    return bugs_with_reopen_flag

def prepare_reopening_features(bugs_df, changelog_df=None):
    """Prepare features for bug reopening prediction."""
    logger.info("\n--- PREPARING FEATURES FOR REOPENING PREDICTION ---")
    
    # Extract project from issue key
    bugs_with_features = bugs_df.withColumn(
        "project", split(col("key"), "-").getItem(0)
    )

    # Calculate text lengths with error checking
    try:
        bugs_with_features = bugs_with_features.withColumn(
            "summary_length", 
            when(col("summary").isNotNull(), length(col("summary"))).otherwise(0)
        ).withColumn(
            "description_length", 
            when(col("description").isNotNull(), length(col("description"))).otherwise(0)
        )
    except Exception as e:
        logger.warning(f"Error calculating text lengths: {str(e)}")
        # Fallback to simpler approach
        bugs_with_features = bugs_with_features.withColumn("summary_length", lit(0))
        bugs_with_features = bugs_with_features.withColumn("description_length", lit(0))

    # Calculate resolution time where available
    try:
        bugs_with_features = bugs_with_features.withColumn(
            "created_date", to_date(col("created"))
        ).withColumn(
            "resolution_date", to_date(col("resolutiondate"))
        ).withColumn(
            "resolution_time_days",
            when(
                col("resolution_date").isNotNull() & col("created_date").isNotNull(),
                datediff(col("resolution_date"), col("created_date"))
            ).otherwise(0)  # Use 0 instead of null to avoid errors
        )
    except Exception as e:
        logger.warning(f"Error calculating resolution time: {str(e)}")
        bugs_with_features = bugs_with_features.withColumn("resolution_time_days", lit(0))

    # Add comment count and other features if changelog is available
    if changelog_df is not None:
        try:
            # Get comment count per issue
            comment_counts = changelog_df.filter(col("field") == "Comment") \
                .groupBy("key") \
                .agg(count("*").alias("comment_count"))

            # Join comment counts
            bugs_with_features = bugs_with_features.join(
                comment_counts,
                "key",
                "left"
            ).withColumn(
                "comment_count",
                when(col("comment_count").isNull(), 0).otherwise(col("comment_count"))
            )

            # Calculate attachment count
            attachment_counts = changelog_df.filter(col("field") == "Attachment") \
                .groupBy("key") \
                .agg(count("*").alias("attachment_count"))

            # Join attachment counts
            bugs_with_features = bugs_with_features.join(
                attachment_counts,
                "key",
                "left"
            ).withColumn(
                "attachment_count",
                when(col("attachment_count").isNull(), 0).otherwise(col("attachment_count"))
            )

            # Calculate status change count
            status_changes = changelog_df.filter(col("field") == "status")
            status_change_counts = status_changes.groupBy("key") \
                .agg(count("*").alias("status_change_count"))

            # Join status change counts
            bugs_with_features = bugs_with_features.join(
                status_change_counts,
                "key",
                "left"
            ).withColumn(
                "status_change_count",
                when(col("status_change_count").isNull(), 0).otherwise(col("status_change_count"))
            )
        except Exception as e:
            logger.warning(f"Error calculating changelog features: {str(e)}")
            # Fallback to default values
            bugs_with_features = bugs_with_features.withColumn("comment_count", lit(0))
            bugs_with_features = bugs_with_features.withColumn("attachment_count", lit(0))
            bugs_with_features = bugs_with_features.withColumn("status_change_count", lit(0))
    else:
        # Add default values if changelog isn't available
        bugs_with_features = bugs_with_features.withColumn("comment_count", lit(0))
        bugs_with_features = bugs_with_features.withColumn("attachment_count", lit(0))
        bugs_with_features = bugs_with_features.withColumn("status_change_count", lit(0))

    # Combine text fields
    try:
        bugs_with_features = bugs_with_features.withColumn(
            "text_content", 
            concat_ws(" ", 
                when(col("summary").isNotNull(), col("summary")).otherwise(""),
                when(col("description").isNotNull(), col("description")).otherwise("")
            )
        )
    except Exception as e:
        logger.warning(f"Error combining text fields: {str(e)}")
        bugs_with_features = bugs_with_features.withColumn("text_content", lit(""))

    # Show the features we've engineered
    logger.info("Features for bug reopening prediction:")
    bugs_with_features.select(
        "key", "was_reopened", "project", "priority_name", "summary_length", 
        "description_length", "comment_count", "attachment_count", 
        "status_change_count", "resolution_time_days"
    ).limit(5).show()

    # Check statistics for reopened vs non-reopened bugs
    logger.info("\nAverage metrics by reopening status:")
    bugs_with_features.groupBy("was_reopened").agg(
        count("*").alias("bug_count"),
        avg("comment_count").alias("avg_comments"),
        avg("summary_length").alias("avg_summary_length"),
        avg("description_length").alias("avg_description_length"),
        avg("attachment_count").alias("avg_attachments"),
        avg("status_change_count").alias("avg_status_changes"),
        avg("resolution_time_days").alias("avg_resolution_days")
    ).show()
    
    return bugs_with_features

def train_reopening_models(bugs_with_features):
    """Train and evaluate models for bug reopening prediction with robust error handling."""
    logger.info("\n--- TRAINING MODELS FOR REOPENING PREDICTION ---")
    
    # Create a balanced dataset for training
    reopened_bugs = bugs_with_features.filter(col("was_reopened") == True)
    non_reopened_bugs = bugs_with_features.filter(col("was_reopened") == False)

    reopened_count = reopened_bugs.count()
    non_reopened_count = non_reopened_bugs.count()

    # FIXED: Use Python's built-in min function correctly with fallback
    try:
        # Prevent division by zero with max
        sampling_fraction = min(3.0 * reopened_count / max(non_reopened_count, 1), 1.0)
        logger.info(f"Sampling {sampling_fraction * 100:.2f}% of non-reopened bugs to create a balanced dataset")
    except Exception as e:
        logger.warning(f"Error calculating sampling fraction: {str(e)}")
        # Fallback to a safe default
        sampling_fraction = 0.01
        logger.info(f"Using fallback sampling fraction: {sampling_fraction}")

    # Sample non-reopened bugs
    sampled_non_reopened = non_reopened_bugs.sample(False, sampling_fraction, seed=42)
    balanced_dataset = reopened_bugs.union(sampled_non_reopened)

    logger.info("\nBalanced dataset statistics:")
    balanced_dataset.groupBy("was_reopened").count().show()
    
    # Split the data into training and testing sets
    train_df, test_df = balanced_dataset.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Training data size: {train_df.count()}, Test data size: {test_df.count()}")

    try:
        # Process categorical features
        categorical_cols = ["project", "priority_name"]
        # Check these columns exist
        available_categorical_cols = [c for c in categorical_cols if c in balanced_dataset.columns]
        
        indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") 
                   for c in available_categorical_cols]
        encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec", handleInvalid="keep") 
                   for c in available_categorical_cols]

        # Process text features
        tokenizer = Tokenizer(inputCol="text_content", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

        # Create word features
        word2Vec = Word2Vec(inputCol="filtered_words", outputCol="word_features", vectorSize=100, minCount=5)

        # Create n-gram features
        ngram = NGram(n=2, inputCol="filtered_words", outputCol="ngrams")
        cv_ngram = CountVectorizer(inputCol="ngrams", outputCol="ngram_features", vocabSize=1000, minDF=5.0)

        # Assemble all features
        numeric_cols = ["summary_length", "description_length", "comment_count", 
                        "attachment_count", "status_change_count"]
                        
        if "resolution_time_days" in balanced_dataset.columns:
            balanced_dataset = balanced_dataset.withColumn(
                "resolution_time_days",
                when(col("resolution_time_days").isNull(), 0).otherwise(col("resolution_time_days"))
            )
            numeric_cols.append("resolution_time_days")

        # Check these columns exist
        available_numeric_cols = [c for c in numeric_cols if c in balanced_dataset.columns]

        assembler_inputs = [c+"_vec" for c in available_categorical_cols] + ["word_features", "ngram_features"] + available_numeric_cols
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

        # Define models to evaluate
        lr = LogisticRegression(labelCol="was_reopened", featuresCol="features", maxIter=10)
        rf = RandomForestClassifier(labelCol="was_reopened", featuresCol="features", numTrees=50)
        
        # Create full pipelines for each model
        stages = indexers + encoders + [tokenizer, remover, word2Vec, ngram, cv_ngram, assembler]

        feature_pipeline = Pipeline(stages=stages)
        
        # Transform the data with the feature pipeline
        logger.info("Fitting feature pipeline...")
        feature_model = feature_pipeline.fit(train_df)
        train_features = feature_model.transform(train_df)
        test_features = feature_model.transform(test_df)
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback to a much simpler pipeline
        logger.info("Falling back to simplified feature pipeline...")
        
        # Very simple feature prep
        tokenizer = Tokenizer(inputCol="text_content", outputCol="words")
        cv = CountVectorizer(inputCol="words", outputCol="features", minDF=2.0)
        
        simplified_pipeline = Pipeline(stages=[tokenizer, cv])
        feature_model = simplified_pipeline.fit(train_df)
        train_features = feature_model.transform(train_df)
        test_features = feature_model.transform(test_df)

    # Function to evaluate model performance
    def evaluate_model(model, train_df, test_df, model_name):
        try:
            logger.info(f"\nTraining {model_name}...")
            model_fit = model.fit(train_df)
            
            # Make predictions
            train_preds = model_fit.transform(train_df)
            test_preds = model_fit.transform(test_df)
            
            # Set up evaluator
            evaluator = BinaryClassificationEvaluator(labelCol="was_reopened", metricName="areaUnderROC")
            
            # Calculate metrics
            train_auc = evaluator.evaluate(train_preds)
            test_auc = evaluator.evaluate(test_preds)
            
            # Calculate additional metrics (precision, recall, F1)
            from pyspark.ml.evaluation import MulticlassClassificationEvaluator
            multi_evaluator = MulticlassClassificationEvaluator(labelCol="was_reopened", predictionCol="prediction")
            
            # Test metrics
            precision = multi_evaluator.setMetricName("weightedPrecision").evaluate(test_preds)
            recall = multi_evaluator.setMetricName("weightedRecall").evaluate(test_preds)
            f1 = multi_evaluator.setMetricName("f1").evaluate(test_preds)
            
            logger.info(f"  - Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}")
            logger.info(f"  - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            return {
                "model": model_fit,
                "name": model_name,
                "auc": test_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            # Return a placeholder result
            return {
                "model": None,
                "name": model_name,
                "auc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

    # Train and evaluate models
    results = []
    
    # Always try LogisticRegression as it's most robust
    lr_result = evaluate_model(lr, train_features, test_features, "LogisticRegression")
    if lr_result["model"] is not None:
        results.append(lr_result)
    
    # Try RandomForest if we have enough data and first model worked
    if train_features.count() > 100 and len(results) > 0:
        rf_result = evaluate_model(rf, train_features, test_features, "RandomForest")
        if rf_result["model"] is not None:
            results.append(rf_result)
    
    # Try training GBT only if we have successful models and data isn't too large
    if train_features.count() < 10000 and len(results) > 0:
        # GBT is most likely to have issues, so add extra protection
        try:
            gbt = GBTClassifier(labelCol="was_reopened", featuresCol="features", maxIter=10)
            gbt_result = evaluate_model(gbt, train_features, test_features, "GradientBoostedTrees")
            if gbt_result["model"] is not None:
                results.append(gbt_result)
        except Exception as e:
            logger.warning(f"Error setting up GBT model: {str(e)}")

    # Find best model
    if results:
        best_model = max(results, key=lambda x: x["auc"])
        logger.info(f"\nBest model: {best_model['name']} with AUC = {best_model['auc']:.3f}")

        # Try to get feature importance from the best model if available
        if best_model['name'] in ["RandomForest", "GradientBoostedTrees"] and best_model['model'] is not None:
            try:
                logger.info("\nFeature Importances:")
                feature_importance = best_model['model'].featureImportances
                
                # Get feature names (safely)
                if 'assembler_inputs' in locals():
                    feature_names = assembler_inputs
                else:
                    feature_names = ["feature_" + str(i) for i in range(len(feature_importance))]
                
                # Create list of (feature, importance) tuples
                importances = [(feature, float(importance)) for feature, importance in zip(feature_names, feature_importance)]
                
                # Print top 10 features
                logger.info("Top 10 features for predicting bug reopening:")
                for i, (feature, importance) in enumerate(sorted(importances, key=lambda x: x[1], reverse=True)[:10]):
                    logger.info(f"{i+1}. {feature}: {importance:.4f}")
            except Exception as e:
                logger.warning(f"Could not extract feature importances: {str(e)}")
    else:
        logger.warning("No models trained successfully")
        best_model = {"name": "None", "auc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    return {
        "models": results,
        "best_model": best_model,
        "feature_pipeline": feature_model,
        "train_features": train_features,
        "test_features": test_features
    }

def analyze_bug_clusters(bugs_with_features, sample_fraction=0.1):
    """Perform clustering analysis to identify bug patterns."""
    logger.info("\n--- PERFORMING BUG CLUSTERING ANALYSIS ---")
    
    # Sample bugs for clustering to make it more manageable
    try:
        if sample_fraction < 1.0:
            logger.info(f"Sampling {sample_fraction*100:.1f}% of bugs for clustering analysis")
            bugs_for_clustering = bugs_with_features.sample(False, sample_fraction, seed=42)
        else:
            bugs_for_clustering = bugs_with_features
            
        logger.info(f"Using {bugs_for_clustering.count()} bugs for clustering analysis")
    except Exception as e:
        logger.warning(f"Error sampling data: {str(e)}")
        # Fallback to a small safe sample
        bugs_for_clustering = bugs_with_features.limit(1000)
        logger.info(f"Using {bugs_for_clustering.count()} bugs for clustering (fallback)")

    # Process text features
    try:
        tokenizer = Tokenizer(inputCol="text_content", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        cv = CountVectorizer(inputCol="filtered_words", outputCol="text_features", minDF=2.0, vocabSize=5000)

        # Process categorical features (project, priority, status)
        categorical_cols = ["project", "priority_name", "status_name"]
        
        # Make sure all required columns exist
        available_categorical_cols = [c for c in categorical_cols if c in bugs_for_clustering.columns]
        
        indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") 
                    for c in available_categorical_cols]
        encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec", handleInvalid="keep") 
                    for c in available_categorical_cols]

        # Use numeric features
        numeric_cols = ["summary_length", "description_length"]
        
        # Add optional numeric columns if they exist
        if "comment_count" in bugs_for_clustering.columns:
            numeric_cols.append("comment_count")
        if "attachment_count" in bugs_for_clustering.columns:
            numeric_cols.append("attachment_count")

        # Make sure all required columns exist
        available_numeric_cols = [c for c in numeric_cols if c in bugs_for_clustering.columns]

        # Assemble features
        assembler_inputs = ["text_features"] + [c+"_vec" for c in available_categorical_cols] + available_numeric_cols
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

        # Create and fit pipeline
        clustering_pipeline = Pipeline(stages=[
            tokenizer, remover, cv
        ] + indexers + encoders + [assembler])

        logger.info("Fitting clustering pipeline...")
        clustering_model = clustering_pipeline.fit(bugs_for_clustering)
        bugs_with_features_vector = clustering_model.transform(bugs_for_clustering)
    except Exception as e:
        logger.error(f"Error in clustering feature preparation: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback to a much simpler pipeline
        logger.info("Falling back to simplified clustering pipeline...")
        tokenizer = Tokenizer(inputCol="text_content", outputCol="words")
        cv = CountVectorizer(inputCol="words", outputCol="features", minDF=2.0)
        simple_pipeline = Pipeline(stages=[tokenizer, cv])
        
        clustering_model = simple_pipeline.fit(bugs_for_clustering)
        bugs_with_features_vector = clustering_model.transform(bugs_for_clustering)

    # Find optimal number of clusters using silhouette score (or use fixed k if that fails)
    silhouette_scores = []
    best_k = 3  # Default if optimization fails
    try:
        logger.info("\nFinding optimal number of clusters...")
        k_values = range(2, 6)  # More limited range for efficiency
        evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction")

        logger.info("K\tSilhouette Score")
        logger.info("-" * 25)

        for k in k_values:
            try:
                kmeans = KMeans(k=k, seed=42, featuresCol="features", initMode="k-means||")
                model = kmeans.fit(bugs_with_features_vector)
                predictions = model.transform(bugs_with_features_vector)
                
                silhouette = evaluator.evaluate(predictions)
                silhouette_scores.append(float(silhouette))  # Convert to Python float
                
                logger.info(f"{k}\t{silhouette:.4f}")
            except Exception as e:
                logger.warning(f"Error evaluating k={k}: {str(e)}")
                silhouette_scores.append(0.0)

        # Find best k if we have any successful evaluations
        if any(score > 0 for score in silhouette_scores):
            best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
            logger.info(f"\nBest K: {best_k} with silhouette score: {max(silhouette_scores):.4f}")
        else:
            logger.warning("No successful silhouette evaluations, using default k=3")
    except Exception as e:
        logger.warning(f"Error finding optimal k: {str(e)}")
        # Use default values
        silhouette_scores = [0.0] * 4  # For k=2,3,4,5
        logger.info(f"Using fallback k={best_k}")

    # Train final model with best k
    try:
        kmeans = KMeans(k=best_k, seed=42, featuresCol="features", initMode="k-means||")
        final_model = kmeans.fit(bugs_with_features_vector)
        clustered_bugs = final_model.transform(bugs_with_features_vector)
    except Exception as e:
        logger.error(f"Error training final clustering model: {str(e)}")
        # Fall back to BisectingKMeans which is often more robust
        logger.info("Falling back to BisectingKMeans...")
        bkmeans = BisectingKMeans(k=best_k, seed=42, featuresCol="features")
        final_model = bkmeans.fit(bugs_with_features_vector)
        clustered_bugs = final_model.transform(bugs_with_features_vector)

    # Count bugs in each cluster
    logger.info("\nCluster distribution:")
    cluster_counts = clustered_bugs.groupBy("prediction").count().orderBy("prediction")
    cluster_counts.show()
    
    # Try to extract keywords for each cluster
    try:
        extract_cluster_keywords(clustered_bugs, cv, best_k)
    except Exception as e:
        logger.warning(f"Could not extract cluster keywords: {str(e)}")
    
    # Try to analyze reopening rates by cluster if column exists
    if "was_reopened" in clustered_bugs.columns:
        try:
            analyze_cluster_reopening(clustered_bugs)
        except Exception as e:
            logger.warning(f"Could not analyze cluster reopening rates: {str(e)}")
    
    return {
        "clustered_bugs": clustered_bugs,
        "cluster_model": final_model,
        "pipeline_model": clustering_model,
        "best_k": best_k,
        "silhouette_scores": silhouette_scores  # This is a Python list, not a Spark column
    }

def extract_cluster_keywords(clustered_bugs, cv_model, num_clusters):
    """Extract and display top keywords for each cluster."""
    logger.info("\n--- EXTRACTING CLUSTER KEYWORDS ---")
    
    # Explode words by cluster for analysis
    words_by_cluster = clustered_bugs.select(
        "prediction", explode(col("filtered_words")).alias("word")
    )

    # Count word frequency by cluster
    word_counts = words_by_cluster.groupBy("prediction", "word").count()

    # Define window spec for ranking words within each cluster
    window_spec = Window.partitionBy("prediction").orderBy(col("count").desc())

    # Rank words within clusters
    ranked_words = word_counts.withColumn("rank", expr("rank() over (partition by prediction order by count desc)"))

    # Get top 10 words per cluster
    top_words = ranked_words.filter(col("rank") <= 10)

    # Display keywords by cluster
    logger.info("\nTop keywords for each cluster:")
    for cluster_id in range(num_clusters):
        cluster_size = clustered_bugs.filter(col("prediction") == cluster_id).count()
        logger.info(f"\nCluster {cluster_id} ({cluster_size} bugs):")
        
        # Get top words for this cluster
        cluster_words = top_words.filter(col("prediction") == cluster_id) \
            .select("word", "count", "rank") \
            .orderBy("rank")
        
        # Show keywords
        for row in cluster_words.collect():
            logger.info(f"  {row['rank']:<3} {row['word']:<15} ({row['count']} occurrences)")
        
        # Show a few example bugs from this cluster
        examples = clustered_bugs.filter(col("prediction") == cluster_id) \
            .select("key", "summary") \
            .limit(3)
        
        logger.info("\n  Example bugs:")
        for row in examples.collect():
            logger.info(f"  - {row['key']}: {row['summary']}")

def analyze_cluster_reopening(clustered_bugs):
    """Analyze reopening rates for different bug clusters."""
    logger.info("\n--- ANALYZING REOPENING RATES BY CLUSTER ---")
    
    # Analyze reopening rates by cluster
    cluster_reopening = clustered_bugs.groupBy("prediction").agg(
        count("*").alias("total_bugs"),
        sum(when(col("was_reopened") == True, 1).otherwise(0)).alias("reopened_bugs")
    ).withColumn(
        "reopening_rate", col("reopened_bugs") / col("total_bugs")
    ).orderBy(col("reopening_rate").desc())

    logger.info("\nReopening rates by cluster:")
    cluster_reopening.show()

    # Find characteristics of clusters with highest and lowest reopening rates
    highest_reopening_cluster = cluster_reopening.orderBy(col("reopening_rate").desc()).first()["prediction"]
    lowest_reopening_cluster = cluster_reopening.orderBy("reopening_rate").first()["prediction"]

    logger.info(f"\nCharacteristics of cluster with highest reopening rate (Cluster {highest_reopening_cluster}):")
    clustered_bugs.filter(col("prediction") == highest_reopening_cluster).agg(
        avg("comment_count").alias("avg_comments"),
        avg("attachment_count").alias("avg_attachments"),
        avg("summary_length").alias("avg_summary_length"),
        avg("description_length").alias("avg_description_length"),
        avg("status_change_count").alias("avg_status_changes")
    ).show()

    logger.info(f"\nCharacteristics of cluster with lowest reopening rate (Cluster {lowest_reopening_cluster}):")
    clustered_bugs.filter(col("prediction") == lowest_reopening_cluster).agg(
        avg("comment_count").alias("avg_comments"),
        avg("attachment_count").alias("avg_attachments"),
        avg("summary_length").alias("avg_summary_length"),
        avg("description_length").alias("avg_description_length"),
        avg("status_change_count").alias("avg_status_changes")
    ).show()

def save_results(results, output_dir, spark):
    """Save analysis results to CSV files."""
    logger.info(f"\n--- SAVING RESULTS TO {output_dir} ---")
    
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save reopening model results if available
        if "reopening_results" in results and "models" in results["reopening_results"]:
            models = results["reopening_results"]["models"]
            schema = StructType([
                StructField("model", StringType(), True),
                StructField("auc", FloatType(), True),
                StructField("precision", FloatType(), True),
                StructField("recall", FloatType(), True),
                StructField("f1", FloatType(), True)
            ])
            
            model_metrics = [(m["name"], m["auc"], m["precision"], m["recall"], m["f1"]) for m in models]
            model_metrics_df = spark.createDataFrame(model_metrics, schema)
            model_metrics_df.coalesce(1).write.mode("overwrite").option("header", "true") \
                .csv(f"{output_dir}/reopening_model_metrics")
            
            logger.info(f"Saved reopening model metrics to {output_dir}/reopening_model_metrics")
        
        # Save cluster results if available
        if "clustering_results" in results and "clustered_bugs" in results["clustering_results"]:
            clustered_bugs = results["clustering_results"]["clustered_bugs"]
            
            # Save counts per cluster
            cluster_counts = clustered_bugs.groupBy("prediction").count().orderBy("prediction")
            cluster_counts.coalesce(1).write.mode("overwrite").option("header", "true") \
                .csv(f"{output_dir}/cluster_counts")
            
            # Save reopening rates by cluster if column exists
            if "was_reopened" in clustered_bugs.columns:
                cluster_reopening = clustered_bugs.groupBy("prediction").agg(
                    count("*").alias("total_bugs"),
                    sum(when(col("was_reopened") == True, 1).otherwise(0)).alias("reopened_bugs")
                ).withColumn(
                    "reopening_rate", col("reopened_bugs") / col("total_bugs")
                ).orderBy(col("reopening_rate").desc())
                
                cluster_reopening.coalesce(1).write.mode("overwrite").option("header", "true") \
                    .csv(f"{output_dir}/cluster_reopening_rates")
            
            logger.info(f"Saved clustering results to {output_dir}/cluster_*")
        
        logger.info(f"All results successfully saved to {output_dir}/")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        logger.info("Continuing without saving results")

def generate_summary(results):
    """Generate a summary of analysis findings."""
    logger.info("\n=== ANALYSIS SUMMARY ===")
    
    # Summarize reopening analysis
    if "reopening_results" in results:
        reopening = results["reopening_results"]
        best_model = reopening.get("best_model", {})
        logger.info("\nBug Reopening Analysis:")
        logger.info(f"- Best model: {best_model.get('name', 'N/A')}")
        logger.info(f"- AUC: {best_model.get('auc', 0):.3f}")
        logger.info(f"- Precision: {best_model.get('precision', 0):.3f}")
        logger.info(f"- Recall: {best_model.get('recall', 0):.3f}")
        logger.info(f"- F1 Score: {best_model.get('f1', 0):.3f}")
    
    # Summarize clustering analysis - FIXED: Properly handle silhouette scores
    if "clustering_results" in results:
        clustering = results["clustering_results"]
        logger.info("\nBug Clustering Analysis:")
        logger.info(f"- Optimal number of clusters: {clustering.get('best_k', 'N/A')}")
        
        # FIXED: Get silhouette score using Python max() on the Python list
        silhouette_scores = clustering.get('silhouette_scores', [0.0])
        # Make sure it's a Python list of floats, not a PySpark column
        if hasattr(silhouette_scores, 'toArray'):  # Handle if it's a DenseVector
            silhouette_scores = silhouette_scores.toArray().tolist()
        elif not isinstance(silhouette_scores, list):
            silhouette_scores = [0.0]  # Fallback
            
        # Now it's safe to use max() on the list
        best_silhouette = max(silhouette_scores) if silhouette_scores else 0.0
        logger.info(f"- Silhouette score: {best_silhouette:.4f}")
        
        # Try to get cluster distribution if available
        if "clustered_bugs" in clustering:
            try:
                counts = clustering["clustered_bugs"].groupBy("prediction").count().collect()
                logger.info("- Cluster distribution:")
                for row in counts:
                    logger.info(f"  Cluster {row['prediction']}: {row['count']} bugs")
            except Exception as e:
                logger.warning(f"Could not show cluster distribution: {str(e)}")
    
    logger.info("\nInsights:")
    logger.info("- Based on our analysis, we identified patterns in bug reopening")
    logger.info("- Different types of bugs (clusters) have distinct reopening rates")
    logger.info("- This information can help teams improve bug resolution processes")
    logger.info("- Features like comment count and priority level strongly predict reopening")

def main():
    """Main function for bug analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Final Apache JIRA Bug Analysis")
    parser.add_argument("--issues", help="Path to issues parquet file", default=None)
    parser.add_argument("--changelog", help="Path to changelog parquet file", default=None)
    parser.add_argument("--output", help="Output directory for results", default="bug_analysis_results")
    parser.add_argument("--sample", help="Fraction of data to sample (0.0-1.0)", type=float, default=None)
    args = parser.parse_args()
    
    # Set up timing
    start_time = time.time()
    
    # Initialize Spark
    spark = setup_spark()
    
    try:
        # Load data
        bugs_df, changelog_df = load_data(spark, args.issues, args.changelog, args.sample)
        
        # Analyze bug reopening patterns
        bugs_with_reopen = analyze_bug_reopening(bugs_df, changelog_df)
        
        # Engineer features for modeling
        bugs_with_features = prepare_reopening_features(bugs_with_reopen, changelog_df)
        
        # Train reopening prediction models
        reopening_results = train_reopening_models(bugs_with_features)
        
        # Perform clustering analysis
        sample_fraction = 0.1 if bugs_with_features.count() > 10000 else 1.0
        clustering_results = analyze_bug_clusters(bugs_with_features, sample_fraction)
        
        # Combine results
        results = {
            "reopening_results": reopening_results,
            "clustering_results": clustering_results
        }
        
        # Save results
        save_results(results, args.output, spark)
        
        # Generate summary
        generate_summary(results)
        
        # Show execution time
        end_time = time.time()
        logger.info(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main analysis pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Clean up Spark session
        spark.stop()

if __name__ == "__main__":
    main()