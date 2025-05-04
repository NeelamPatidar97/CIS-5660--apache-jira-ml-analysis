from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import unix_timestamp, col, round, current_timestamp, expr, to_timestamp
from pyspark.sql import DataFrame

# Create Spark session
spark = SparkSession.builder.appName("ApacheJiraML").getOrCreate()

# Fix for Spark 3+ timestamp parsing issues
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Function to clean column names
def clean_column_names(df: DataFrame) -> DataFrame:
    for col_name in df.columns:
        clean_name = col_name.replace(".", "_").replace(" ", "_")
        if clean_name != col_name:
            df = df.withColumnRenamed(col_name, clean_name)
    return df

# Load CSVs from HDFS
issues_df = spark.read.option("header", True).option("inferSchema", True).option("multiLine", True).option("escape", "\"").csv("/user/your_username/Apache_JIRA_Issues/issues.csv")
changelog_df = spark.read.option("header", True).option("inferSchema", True).option("multiLine", True).option("escape", "\"").csv("/user/your_username/Apache_JIRA_Issues/changelog.csv")
comments_df = spark.read.option("header", True).option("inferSchema", True).option("multiLine", True).option("escape", "\"").csv("/user/your_username/Apache_JIRA_Issues/comments.csv")

# Clean column names
issues_df = clean_column_names(issues_df)
changelog_df = clean_column_names(changelog_df)
comments_df = clean_column_names(comments_df)

# Deduplicate
issues_df = issues_df.dropDuplicates(["key"])
comments_df = comments_df.dropDuplicates(["comment_id"])
changelog_df = changelog_df.dropDuplicates(["id"])

# Drop rows with nulls in required columns
issues_df = issues_df.dropna(subset=["key", "created", "updated", "status_name", "priority_name", "issuetype_name"])
comments_df = comments_df.dropna(subset=["key", "comment_id"])
changelog_df = changelog_df.dropna(subset=["key", "field", "id"])

# Join comment count and status change count
comment_counts = comments_df.groupBy("key").agg(F.count("comment_id").alias("comment_count"))
status_change_counts = changelog_df.filter(col("field") == "status").groupBy("key").agg(F.count("id").alias("status_change_count"))

issues_df = issues_df.join(comment_counts, "key", "left").join(status_change_counts, "key", "left") \
    .fillna({"comment_count": 0, "status_change_count": 0})

# 🔍 Filter out malformed timestamps (e.g., starting with 2-digit year like '12-09-17')
issues_df = issues_df.filter(col("created").startswith("20")).filter(col("updated").startswith("20"))

# Convert to timestamp
issues_df = issues_df \
    .withColumn("created", to_timestamp("created", "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("updated", to_timestamp("updated", "yyyy-MM-dd HH:mm:ss"))


#What is the distribution of issue resolution times for last two years?
# Calculate resolution_days
issues_df = issues_df.withColumn(
    "resolution_days",
    round((unix_timestamp("updated") - unix_timestamp("created")) / 86400, 2)
)

# Filter: resolved issues in last 2 years
two_years_ago = expr("add_months(current_timestamp(), -24)")
resolved_df = issues_df.filter(
    (col("resolution_days").isNotNull()) & (col("created") >= two_years_ago)
)

# Preview
resolved_df.select(
    "key", "created", "updated", "resolution_days",
    "project_name", "status_name", "priority_name", "issuetype_name"
).show(5)


"""
# ✅ Export to HDFS
# Select columns for export
export_df = resolved_df.select(
    "key",
    "created",
    "updated",
    "resolution_days",
    "project_name",
    "status_name",
    "priority_name",
    "issuetype_name",
    "comment_count",
    "status_change_count"
)


# Export as single CSV with header for visualization
export_df.coalesce(1).write.mode("overwrite").option("header", True).csv("/user/npatida/output/resolved_df_q1")
"""


# Question 2 How does resolution time vary across issue priorities and statuses?
from pyspark.sql.functions import avg

# Group by priority and status to get average resolution time
priority_status_avg_df = resolved_df.groupBy("priority_name", "status_name") \
    .agg(avg("resolution_days").alias("avg_resolution_days")) \
    .orderBy("priority_name", "status_name")
    
priority_status_avg_df.show(5)

"""
# Save to HDFS for Tableau or Excel
priority_status_avg_df.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv("/user/npatida/output/q2_priority_status_resolution")
"""

# Q3: Top 10 fields changed most frequently + how many unique issues were affected
changelog_summary_df = changelog_df.groupBy("field") \
    .agg(
        F.count("*").alias("total_changes"),
        F.countDistinct("key").alias("affected_issues")
    ) \
    .orderBy(F.desc("total_changes")) \
    .limit(10)

# Show the summary
changelog_summary_df.show(truncate=False)

# Save to HDFS for reporting
changelog_summary_df.coalesce(1).write.mode("overwrite").option("header", True) \
    .csv("/user/npatida/output/q3_field_change_summary")

