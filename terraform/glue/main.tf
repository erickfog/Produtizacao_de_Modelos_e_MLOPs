resource "aws_glue_catalog_database" "btc_database" {
  name = "btc_data"
}

resource "aws_glue_catalog_table" "btc_table" {
  name          = "btc_prices"
  database_name = aws_glue_catalog_database.btc_database.name

  table_type = "EXTERNAL_TABLE"
  parameters = {
    "classification"  = "json" # Ou "parquet" se estiver usando esse formato
    "compressionType" = "gzip" # Caso use compressão
  }

  storage_descriptor {
    location      = "s3://my-firehose-bucket/data/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      serialization_library = "org.openx.data.jsonserde.JsonSerDe"
    }

    # Definição das colunas
    columns {
      name = "price"
      type = "double"
    }

    columns {
      name = "coleta"
      type = "string"
    }
  }
}

