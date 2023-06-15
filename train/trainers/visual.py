"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import os
import scipy
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import colorcet as cc
from matplotlib.colors import LinearSegmentedColormap
import torch
from spinelib.imgio.plot import imshow


class Visualizer:

    def __init__(self, keys):
        self.wins = {k: None for k in keys}
        self.len_key=len(self.wins)
        self.keys=keys
        n_images=1
        f,axs=plt.subplots(nrows=n_images,ncols=self.len_key,figsize=(10*self.len_key,10))# figsize=(H,W)
        self.fig=f
        for n,key in enumerate(self.keys):
            if n_images==1:
                self.wins[key]=axs[n]
                
            else:
                self.wins[key]=axs[:,n]
    def save(self):
        plt.tight_layout()
        plt.savefig("test.png")
        # plt.close()
        # self.wins = {k: None for k in self.keys}
    def display(self, image, key):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1

        
        
            #self.wins[key] = plt.subplots(nrows=n_images,ncols=self.len_key)

        ax = self.wins[key]
        n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1

        assert n_images == n_axes

        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
           
            if key=="seed":
                nmax=1
                print("====================================",image.max())
                c=ax.imshow(self.prepare_img(image),vmax=nmax)
            elif "pred" in key:
                c=ax.imshow(self.prepare_img(image),cmap="jet")
            elif "image"==key:
                c=ax.imshow(self.prepare_img(image),cmap="gray",vmax=0.7)
            elif key in ["GT","label"]:
                cm=cc.glasbey
                cm[0]="#FFFFFF"
                cmap=LinearSegmentedColormap.from_list("isolum",cm)
         
                c=ax.imshow(self.prepare_img(image),cmap=cmap,interpolation="none")
            elif key=="angle":
                c=ax.imshow(self.prepare_img(image),cmap="twilight")
            else:
                c=ax.imshow(self.prepare_img(image))
            ax.set_title(key)
            #plt.colorbar(c,ax=ax)
        else:
            for i in range(n_images):
                ax[i].cla()
                ax[i].set_axis_off()
                c=ax[i].imshow(self.prepare_img(image[i]),cmap="jet")
                

        # plt.draw()
        # self.mypause(0.001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
                image=image[:,:,::-1]
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


def get_ball_grid():
    
    fig = plt.figure()

    # draw vector
    soa = np.array([[0, 0, 1, 1, -2, 0], [0, 0, 2, 1, 1, 0],
                [0, 0, 3, 2, 1, 0], [0, 0, 4, 0.5, 0.7, 0]])

    X, Y, Z, U, V, W = zip(*soa)
    ax = fig.add_subplot(111, projection='3d')

    # draw sphere
    
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray",alpha=0.5,rstride=2, cstride=2)
    ax.set_zlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.set_xlim(-1.2,1.2)
    ax.set_title("Sphere")
    ax._axis3don = False
    return fig,ax
    plt.show()