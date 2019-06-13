"""
Author: Joseph Tsung
"""

# import necessary libraries
import os
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
# import pydicom
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_similarity_score, mutual_info_score

MAX_DIFF = 128

class ImageSimilarity:
    """ Instantiate a class for storing images and hashes """
    
    def __init__(self, images_dir, processing_func):
        """for each image: hash = hashing_func(processing_func(image))
           save hash in list 
        Parameters:
        -path or directory to the image folder
        -preprocessing function for the images
        """
        
        # necessary placeholder data structures
        self.images_dir = images_dir
        self.processing_func = processing_func
        self.paths = []
        self.hashes = []
        self.phashes = []
        
        # loop through image directory to store images, paths, and hashes
        image_files = os.listdir(images_dir)
        for i in range(len(image_files)):
            if not image_files[i].endswith(".png"):
                continue
            path = os.path.join(images_dir, image_files[i])
            self.paths.append(image_files[i])

            im = Image.open(path)
            im = self.processing_func(np.array(im))
            im = Image.fromarray(im)
     
            h = imagehash.dhash(im)
            self.hashes.append(h)
            ph = imagehash.phash(im)
            self.phashes.append(ph)
    
    def query_image(self, image):
        """ image_hash = hashing_func(processing_func(image))
            find most similar images by hamming distance 
        
        Parameters:
        -query image
        
        Output:
        -final_matches: paths to the similar images
        """

        matches = []
        final_matches = []
        hash_differences = []
        phash_differences = []
    
        query = np.array(image)
        query_h = imagehash.dhash(Image.fromarray(query))
        query_ph = imagehash.phash(Image.fromarray(query))
        
        # generate hash differences
        for i in range(len(self.hashes)):
        
            diff = query_h-self.hashes[i]
            hash_differences.append(diff)
        
        # use k means to find threshold for similarity cutoff
        kmeans = KMeans(n_clusters=2).fit(np.array(hash_differences).reshape(-1,1))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)

        for i in range(len(hash_differences)):
            if hash_differences[i] < threshold:
                matches.append(i)
    
        # if no match, output this message
        if not matches:
            pass
        else:
            # for images that fall within dhash threshold, try nesting phash or Dice, Jaccard, Mutual information
            for j in matches:
                    
                #if image.endswith('.dcm'):
                #    ds = pydicom.dcmread(matches[j])
                #    im = Image.fromarray(ds.pixel_array)     
                
                ph = self.phashes[j]
                diff = query_ph - ph
                phash_differences.append(diff)
            
                if diff < 10:
                    final_matches.append({
                        "name": self.paths[j],
                        "similarity": diff / MAX_DIFF,
                        "response": 1,
                    })
     
        # skip this for now
        #if not final_matches:
        #    final_matches= matches

        # final output: paths to the images
        return final_matches
