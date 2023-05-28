---
creation_date: 2023-01-02 17:29
tags: SOURCE-COURSE-NOTE, STATUS-COMPLETE
---

## Additional Metadata

- **Course Link**:: [databricks-apache-spark-programming](databricks-apache-spark-programming.md)]
- **Course Note URL**:: 
- **Video Link**:: 
	- [DataFrames](https://customer-academy.databricks.com/learn/course/63/play/3894:610/dataframes)
- **RelatedTopics**:: 
	- [102-moc-databricks](102-moc-databricks.md)

---
# Documentation

* Pyspask API - [API Reference — PySpark 3.3.1 documentation](https://spark.apache.org/docs/latest/api/python/reference/index.html)
# General

* Can interact with spark SQL using Dataframe API
* The same query can be expressed via SQL or dataframe API
* SQL engine will create the same query plan
* Dataframe is a distributed collection of data
	* Grouped into name columns
* Schema defines the column names and type of DataFrame, e.g:

```bash
item: string
name: string
price: double
qty: int
```

* DataFrame t**ransformations** are methods that **return a new DataFrame** and laziily evaluated
	* Can chain methods together to build a new DataFrame
* Dataframe actions are **methods** that **trigger computations**, e.g:
	* df.count()
	* df.collect()
	* df.show()
* An action is **needed** to trigger the execution of any DataFrame transformation.

# Spark Session

* Single entry point to all DataFrame API functionality
* Automatically created in Databricks notebooks
* Few methods to create DataFrames:
	* `sql` - returns a DF representing the result of the given query
	* `table` - return the specified table as a DataFrame
	* `read` - Returns a DataFrame reader that can be used to read data in as a DataFrame
	* `range` 
	* `createDataFrame` - Creates a DataFrame from a list of tuples, primarily used for testing.

# Query Execution

* Query Plan
	* Optmised query Plan
* RDD
	* Resilient Distributed Dataset
		* Low level abstraction

# Examples

## Transformations

* Storing DF in `budget_df`
* `schema` and `printSchema`
* 

```python
budget_df = (spark
             .table("products") # Selecting the products table
             .select("name", "price")
             .where("price < 200")
             .orderBy("price")
            )

display(budget_df)


budget_df.printSchema()

```

## Actions

* `show()` 
	* displays top n rows of DF in tabular form
* `count()` 
	* returns the number of rows
* describe() and summary()
	* Basic statistics
* first()
	* Returns the first row
* head()
	* Returns the last row
* collect()
	* Returns an array that contains all rows in this DF
* take()
	* Returns an array of first n rows

Example
```python
(products_df
  .select("name", "price")
  .where("price < 200")
  .orderBy("price")
  .show())
```

## Convert between DataFrames and SQL

```python
budget_df.createOrReplaceTempView("budget")
display(spark.sql("SELECT * FROM budget"))
```


# DataFrame  Reader

[apache-spark-programming-with-databricks/ASP 2.2 - Reader & Writer.py at published · databricks-academy/apache-spark-programming-with-databricks · GitHub](https://github.com/databricks-academy/apache-spark-programming-with-databricks/blob/published/ASP%202%20-%20Spark%20Core/ASP%202.2%20-%20Reader%20%26%20Writer.py)

* Dataframe reader is accessible using `spark.read`

---
## Read CSV

## Automatic Schema

* Separation options
* First line as a header
```python
users_csv_path = f"{DA.paths.datasets}/ecommerce/users/users-500k.csv"

users_df = (spark
           .read
           .option("sep", "\t")
           .option("header", True)
           .option("inferSchema", True)
           .csv(users_csv_path)
          )

users_df.printSchema()
```

Using Spark's Python API
```python
users_df = (spark
           .read
           .csv(users_csv_path, sep="\t", header=True, inferSchema=True)
          )
```


### Manual Schema -  Struct

```python

from pyspark.sql.types import LongType, StringType, StructType, StructField

user_defined_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("user_first_touch_timestamp", LongType(), True),
    StructField("email", StringType(), True)
])
users_df = (spark
           .read
           .option("sep", "\t")
           .option("header", True)
           .schema(user_defined_schema)
           .csv(users_csv_path)
          )


```

### Manual Schema - DDL

```python
ddl_schema = "user_id string, user_first_touch_timestamp long, email string"

users_df = (spark
           .read
           .option("sep", "\t")
           .option("header", True)
           .schema(ddl_schema)
           .csv(users_csv_path)
          )
```

---

## Read JSON

### Automatic Schema

```python
events_json_path = f"{DA.paths.datasets}/ecommerce/events/events-500k.json"

events_df = (spark
            .read
            .option("inferSchema", True)
            .json(events_json_path)
           )

events_df.printSchema()
```

###  Manual Schema using Struct

* Advantage - faster loading

```python
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, LongType, StringType, StructType, StructField

user_defined_schema = StructType([
    StructField("device", StringType(), True),
    StructField("ecommerce", StructType([
        StructField("purchaseRevenue", DoubleType(), True),
        StructField("total_item_quantity", LongType(), True),
        StructField("unique_items", LongType(), True)
    ]), True),
    StructField("event_name", StringType(), True),
    StructField("event_previous_timestamp", LongType(), True),
    StructField("event_timestamp", LongType(), True),
    StructField("geo", StructType([
        StructField("city", StringType(), True),
        StructField("state", StringType(), True)
    ]), True),
    StructField("items", ArrayType(
        StructType([
            StructField("coupon", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("item_name", StringType(), True),
            StructField("item_revenue_in_usd", DoubleType(), True),
            StructField("price_in_usd", DoubleType(), True),
            StructField("quantity", LongType(), True)
        ])
    ), True),
    StructField("traffic_source", StringType(), True),
    StructField("user_first_touch_timestamp", LongType(), True),
    StructField("user_id", StringType(), True)
])

events_df = (spark
            .read
            .schema(user_defined_schema)
            .json(events_json_path)
           )

```

# DataFrame Writer

## Write to Parquet

```python
users_output_dir = f"{DA.paths.working_dir}/users.parquet"

(users_df
 .write
 .option("compression", "snappy")
 .mode("overwrite")
 .parquet(users_output_dir)
)

# Verify
display(
    dbutils.fs.ls(users_output_dir)
)

# Additional method

(users_df
 .write
 .parquet(users_output_dir, compression="snappy", mode="overwrite")
)

```

## Write to a global table

```python
events_df.write.mode("overwrite").saveAsTable("events")
```


## Write to DeltaLake

```python
events_output_path = f"{DA.paths.working_dir}/delta/events"

(events_df
 .write
 .format("delta")
 .mode("overwrite")
 .save(events_output_path)
)
```


# DataFrames and Columns and Rows

* Can't manipulate a column outside of context of DataFrame
	* Can only be transformed

## Accessing columns

```python
df['columnName']
df.columnName

# Refers ti generic column
col("columnName")
col("columnName.field")
```

## Create a column from an expression

```python
col("a") + col("b")
col("a").desc()
col("a").cast("int")*100
```


## Rows

* Row Methods
	* index
	* count
	* asDict
	* row.key
	* row['key']
	* key in row

# Examples working with columns
[apache-spark-programming-with-databricks/ASP 2.3 - DataFrame & Column.py at published · databricks-academy/apache-spark-programming-with-databricks · GitHub](https://github.com/databricks-academy/apache-spark-programming-with-databricks/blob/published/ASP%202%20-%20Spark%20Core/ASP%202.3%20-%20DataFrame%20%26%20Column.py)

```python

from pyspark.sql.functions import col

events_df = spark.read.format("delta").load(DA.paths.events)

# Column expressions
print(col("device"))
col("ecommerce.purchase_revenue_in_usd") + col("ecommerce.total_item_quantity")
col("event_timestamp").desc()
(col("ecommerce.purchase_revenue_in_usd") * 100).cast("int")

# Example of using columns in context of DF
rev_df = (events_df
         .filter(col("ecommerce.purchase_revenue_in_usd").isNotNull())
         .withColumn("purchase_revenue", (col("ecommerce.purchase_revenue_in_usd") * 100).cast("int"))
         .withColumn("avg_purchase_revenue", col("ecommerce.purchase_revenue_in_usd") / col("ecommerce.total_item_quantity"))
         .sort(col("avg_purchase_revenue").desc())
        )

```


## Adding or replace columns

* Returns a new DF with the new column

```python
# New column "mobile", based on device
mobile_df = events_df.withColumn("mobile", col("device").isin("iOS", "Android"))


purchase_quantity_df = events_df.withColumn("purchase_quantity", col("ecommerce.total_item_quantity").cast("int"))
```

## Renaming columns

```python
location_df = events_df.withColumnRenamed("geo", "location")
```

## Filter using SQL expression

```python
purchases_df = events_df.filter("ecommerce.total_item_quantity > 0")

revenue_df = events_df.filter(col("ecommerce.purchase_revenue_in_usd").isNotNull())

android_df = events_df.filter((col("traffic_source") != "direct") & (col("device") == "Android"))
```

## Drop duplicates


```python
distinct_users_df = events_df.dropDuplicates(["user_id"])
```

## Sorting

```python
increase_timestamps_df = events_df.sort("event_timestamp")

# Descending sort
decrease_timestamp_df = events_df.sort(col("event_timestamp").desc())

# Sort and create an alias
increase_sessions_df = events_df.orderBy(["user_first_touch_timestamp", "event_timestamp"])

# Sort on multiple columns
decrease_sessions_df = events_df.sort(col("user_first_touch_timestamp").desc(), col("event_timestamp"))
```