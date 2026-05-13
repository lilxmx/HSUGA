import pickle
import os
from sklearn.decomposition import PCA
# cd data 
# python pca.py
def process_dataset(dataset):
    """
    Apply PCA dimensionality reduction to the specified dataset
    
    Args:
        dataset: Dataset name (beauty/steam/fashion)
    """
    print(f"Processing dataset: {dataset}")
    
    # Load item embeddings
    item_emb_path = os.path.join(dataset, "handled", "itm_emb_np.pkl")
    print(f"Loading item embeddings from {item_emb_path}")
    llm_item_emb = pickle.load(open(item_emb_path, "rb"))
    print(f"Item embeddings shape: {llm_item_emb.shape}")
    
    # Apply PCA dimensionality reduction to 64 dimensions
    print("Applying PCA (n_components=64)...")
    pca = PCA(n_components=64)
    pca_item_emb = pca.fit_transform(llm_item_emb)
    print(f"PCA result shape: {pca_item_emb.shape}")
    
    # Save results
    output_path = os.path.join(dataset, "handled", "pca64_itm_emb_np.pkl")
    print(f"Saving to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(pca_item_emb, f)
    
    print(f"Dataset {dataset} processed successfully!\n")

if __name__ == "__main__":
    # Process three datasets sequentially
    datasets = ["beauty", "steam", "fashion"]
    
    for dataset in datasets:
        try:
            process_dataset(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}\n")
            continue
    
    print("All datasets processed!")
