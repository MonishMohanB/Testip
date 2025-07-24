import time

start_time = time.time()

# Build a dictionary where keys are index labels, and values are row dicts
data_dict = {
    key: {col: getattr(model, col) for col in columns}
    for key, model in cv_result.trainedModels.items()
}

# Create DataFrame directly from dictionary
df = pd.DataFrame.from_dict(data_dict, orient='index')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
