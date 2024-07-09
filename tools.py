import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

common_path = "./chest_Xray"
images_files = os.listdir(common_path)
subfolders = ["train","val","test"]
categories = ["NORMAL","PNEUMONIA"]

def browse_imgs_multi_classes(img_callback, path_folder_callback = None, limit_size = None):
    for subfolder in subfolders:
        for category in categories:
            folder_path = os.path.join(common_path, subfolder, category)
            images_files = os.listdir(folder_path)
            if path_folder_callback is not None:
                path_folder_callback(folder_path, images_files)
            array_limit = limit_size if limit_size is not None else len(images_files)
            for file_name in images_files[:array_limit]:
                if not file_name.endswith(".jpeg"):
                    continue
                image_path = os.path.join(folder_path,file_name)
                img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                img_callback(img, category, file_name)

# Permet de parcourir les images, et pour chaque image, on applique une fonction de callback
# On peut optionnellement appeler une fonction de callback pour chaque dossier
def browse_imgs(img_callback, path_folder_callback = None, limit_size = None):
    for subfolder in subfolders:
        for category in categories:
            # pour avoir tous les chemins des 6 dossiers
            folder_path = os.path.join(common_path, subfolder, category)
            # liste de toutes les images
            images_files = os.listdir(folder_path)
            if path_folder_callback is not None:
                path_folder_callback(folder_path, images_files)
            array_limit = limit_size if limit_size is not None else len(images_files)
            #récupération de toutes les (ou des 'limit_size' premières) images du dossier.
            for file_name in images_files[:array_limit]:
                if not file_name.endswith(".jpeg"):
                    continue
                image_path = os.path.join(folder_path,file_name)
                img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                img_callback(img, category)

def display_distribution_multi_classes(ytrain, testy, label_encoder):
    categories = ['NORMAL', 'VIRUS PNEUMONIA', 'BACTERIA PNEUMONIA']
    # Convert numeric labels back to categorical for display
    ytrain_cat = label_encoder.inverse_transform(ytrain)
    testy_cat = label_encoder.inverse_transform(testy)
    
    test_counts = [np.count_nonzero(testy_cat == category) for category in categories]
    train_counts = [np.count_nonzero(ytrain_cat == category) for category in categories]
    
    # Plot the distribution graphs
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))  # Adjusted subplot count to 2 as only two plots are needed
    ax[0].bar(categories, train_counts)
    ax[0].set_title("Train Distribution")
    ax[1].bar(categories, test_counts)  # Adjusted index to 1 for the second plot
    ax[1].set_title("Test Distribution")
    plt.show()


def display_distribution(ytrain, yval, testy):
    test = (np.count_nonzero(testy == "NORMAL"), np.count_nonzero(testy == "PNEUMONIA"))
    train = (np.count_nonzero(ytrain == "NORMAL"), np.count_nonzero(ytrain == "PNEUMONIA"))
    val = (np.count_nonzero(yval == "NORMAL"), np.count_nonzero(yval == "PNEUMONIA"))
    
    # Tracer les graphiques de distribution
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].bar(["NORMAL", "PNEUMONIA"], train)
    ax[0].set_title("Train Distribution")
    ax[1].bar(["NORMAL", "PNEUMONIA"], val)
    ax[1].set_title("Validation Distribution")
    ax[2].bar(["NORMAL", "PNEUMONIA"], test)
    ax[2].set_title("Test Distribution")
    plt.show()
                
                
def display_imgs(imgs, titles = [], plot_size = (1,1), figsize = (10,8)):
    fig = plt.figure(figsize=figsize)
    index = 0
    for image, title in zip(imgs, titles):
        index += 1
        ax = fig.add_subplot(plot_size[0], plot_size[1], index) 
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        if titles is not None:
            ax.set_title(title)

    plt.tight_layout()
    plt.show()