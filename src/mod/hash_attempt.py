# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:37:14 2019

@author: josep
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

class ImageSimilarity:
    
    def __init__(self, images_dir, processing_func):
        """for each image: hash = hashing_func(processing_func(image))
           save hash in list """
        
            # necessary placeholder data structures
        self.paths = []
        self.hashes = []
        
        #print("%.0f pictures found in root directory" % len(os.listdir(images_dir)))
        
        for i in range(len(os.listdir(images_dir))):
            path = os.path.join(images_dir, os.listdir(images_dir)[i])
            self.paths.append(path)

            #if os.listdir(images_dir)[0].endswith('.dcm'):
                # generate dhash for image
            #    ds = pydicom.dcmread(path)
            #    im = Image.fromarray(ds.pixel_array)     
            if os.listdir(images_dir)[i].endswith('.png'):
                im= Image.open(path)
     
            h = imagehash.dhash(im)
            self.hashes.append(h)
            #print("Getting image: %s \t Generating dhash: %s" % (os.listdir(images_dir)[i], h))
    
    def query_image(self, image):
        """ image_hash = hashing_func(processing_func(image))
            find most similar images by hamming distance or some other metric """

        phashes = []
        matches = []
        final_matches = []
        hash_differences = []
        phash_differences = []
    
        query = np.array(image)
                
        #if image.endswith('.dcm'):
        #    path = os.path.join(images_dir, os.listdir(images_dir)[i])
        #    ds = pydicom.dcmread(path)
        #    query = Image.fromarray(ds.pixel_array)
        
            # Convert to Hounsfield units (HU)
       #     intercept = ds.RescaleIntercept
       #     slope = ds.RescaleSlope
                
       #     if slope != 1:
       #         query = slope * query.astype(np.float64)
       #         query = query.astype(np.int16)
                    
       #         query += np.int16(intercept)
            
        plt.imshow(query, cmap="gray")
        plt.axis("off")
        plt.show()
        
        query_h = imagehash.dhash(Image.fromarray(query))
        query_ph = imagehash.phash(Image.fromarray(query))
        
        #print("Query image dhash is: %s" % query_h)
        #print("Query image phash is: %s \n" % query_ph)

        for i in range(len(self.hashes)):
        
            diff = query_h-self.hashes[i]
            hash_differences.append(diff)

        kmeans = KMeans(n_clusters=2).fit(np.array(hash_differences).reshape(-1,1))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        #print("\n Kmeans threshold: %s \n" % str(threshold))

        for i in range(len(hash_differences)):
            if hash_differences[i] < threshold:
                matches.append(self.paths[i])
                #print("[Match] Image: %s \t hash difference: %s" % (self.paths[i][-13:], hash_differences[i]))
    
        # if no match, output this message
        if not matches:
            #print("No images found")
            pass
        else:
            # for images that fall within dhash threshold, try phash or Dice, Jaccard, Mutual information
            for j in range(len(matches)):
                    
                #if image.endswith('.dcm'):
                #    ds = pydicom.dcmread(matches[j])
                #    im = Image.fromarray(ds.pixel_array)     
                
                im= Image.open(matches[j])
                            
                ph = imagehash.phash(im)
                phashes.append(ph)
                #print("Getting image: %s \t Generating phash: %s" % (matches[j][-13:], ph))
        
                diff = query_ph-ph
                phash_differences.append(diff)
            
                if diff < 10:
                    final_matches.append(matches[j])
                    #print("[Match] Image %s \t phash difference: %s" % (matches[j], phash_differences[j]))
     
        if not final_matches:
            final_matches= matches
        
        for k in range(len(final_matches)):
            
            #if image.endswith('.dcm'):
            #    ds = pydicom.dcmread(final_matches[k])
            #    im = Image.fromarray(ds.pixel_array)
                    
                # Convert to Hounsfield units (HU)
            #    intercept = ds.RescaleIntercept
            #    slope = ds.RescaleSlope
                
            #    if slope != 1:
            #        im = slope * im.astype(np.float64)
            #        im = im.astype(np.int16)
                    
            #        im += np.int16(intercept)
            
            im = Image.open(matches[k])
            im = np.array(im)
                
            plt.imshow(im, cmap="gray")
            plt.axis("off")
            plt.show()
            #print(final_matches[k])
            
            mi = mutual_info_score(query.flatten(), im.flatten())
            jac = jaccard_similarity_score(query.flatten(), im.flatten())
            #print("Mutual information: %.3f \t Jaccard score: %.3f" % (mi, jac))

#def process(img):
#    return img
#
#image = Image.open(r"C:\workspace\223C\Seg2\A08PR4015.png")
#image = np.array(image)
#
#test_dir = r"C:\workspace\223C\Seg2"
#
#test = ImageSimilarity(test_dir, process)
#test.query_image(image)
