import pyarrow as pa
import pyarrow.fs as fs

# Connect to HDFS
hdfs = fs.HadoopFileSystem(host='<namenode-host>', port=9000, user='<your-username>')

# Data to save
data = "Hello, HDFS! This is a test file using pyarrow."

# Save data to HDFS
hdfs_path = '/user/<your-username>/testfile_pyarrow.txt'
with hdfs.open_output_stream(hdfs_path) as writer:
    writer.write(data.encode('utf-8'))

print(f"Data saved to HDFS at: {hdfs_path}")

import pyarrow as pa
import pyarrow.fs as fs

# Load HDFS configuration from hdfs-site.xml
hdfs = fs.HadoopFileSystem(
    host='<namenode-host>',
    port=9000,
    user='<your-username>',
    extra_conf={'hadoop.conf.dir': '/path/to/hadoop/conf'}
)

# Data to save
data = "Hello, HDFS! This is a test file using pyarrow with hdfs-site.xml."

# Save data to HDFS
hdfs_path = '/user/<your-username>/testfile_pyarrow_config.xml'
with hdfs.open_output_stream(hdfs_path) as writer:
    writer.write(data.encode('utf-8'))

print(f"Data saved to HDFS at: {hdfs_path}")
