import healpy as hp
import numpy as np


class Chunkpix:
    """
    Chunkpix stores Healpix maps that are sparse (contain many zeros).
    It uses a lower resolution healpix map that stores which of these
    large pixels, or chunks, are active. Only pixels belonging to active
    chunks need to be stored, all the other ones are assumed to be 0.
    
    Therefore, a chunkpix map consists of chunks of the original map
    and a list of active chunks. This information is enough to reconstruct
    the full original map with no loss of information.
    
    Each instance of this class can store one chunkpix map.
    
    Parameters:
        - nside: resolution of the map
        - nside_chunks: resolution of the low-res chunk map. We suggest
                a default value of 8, corresponding to 768 chunks. But
                depending on your problem and requirements try different
                numbers and check the memory reduction of your maps with
                the method chunk_sparsity. 
        - Data_type_cmap: data type we want to use
        
    The data type to store chunk numbers in active_chunks is np.uint16.
    This can hold integers up to 2**16-1=65535, which would suffice 
    for up to nside_chunks = 64. 
    
    
    Usage:
        1. Initialize class with nside, nside_chunks
        2.a. Add counts one-by-one with increase_ipix_count
        2.b. Or convert a full map to a chunk map with full_map2chunk_map
        3. Retrieve full map with reconstruct_full_map
    """
    
    def __init__(self, nside, nside_chunks=8, data_type_cmap = np.single):
        if(nside_chunks>=nside):
            raise Exception("Please, use nside < nside_chunks")
        
        self.nside = nside
        self.nside_chunks = nside_chunks
        self.data_type_cmap = data_type_cmap
        self.data_type_chunks = np.uint16
        
        self.npix = hp.nside2npix(nside)
        self.nchunks = hp.nside2npix(nside_chunks)
        self.npix_per_chunk = int(self.npix/self.nchunks)
        self.initialize_cmap()
        
        
    def initialize_cmap(self):
        """
        Initialize the chunkpix map to 0
        """
        self.n_active_chunks = 0
        self.active_chunks = np.array([], dtype=self.data_type_chunks)
        self.cmap = np.array([], dtype=self.data_type_cmap)
        print(f"Map intialized to 0. Data type: {self.cmap.dtype}")
        
        
    def print_params(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))
    
    
    def chunk_sparsity(self):
        """
        Return the fraction of memory used by chunkpix compared to a full-size map
        """
        return self.n_active_chunks/self.nchunks
        
    
    def ipix2chunk(self, ipix_full):
        """
        Return the chunk in the full-size map where a pixel belongs to
        """
        return (np.array(ipix_full)//self.npix_per_chunk).astype(self.data_type_chunks)
    
    
    def chunk2ipix_ranges(self, chunk):
        """
        Return a tuple of the pixel ranges belonging to a chunk
        """
        return self.npix_per_chunk * chunk, self.npix_per_chunk * (chunk+1) - 1
    
    
    def increase_ipix_count(self, ipix_full, count_increase=1):
        """
        Increase in cmap the count number corresponding to the pixel ipix_full.
        This operation may involve activating new chunks.
        """
        n_old = self.cmap.sum()
        
        # Chunk this pixel is in
        chunk_value = self.ipix2chunk(ipix_full)
        # Is this chunk active? In what position (key) is it stored?
        chunk_key = np.where(self.active_chunks==chunk_value)[0]

        n = chunk_key.size
        if(n==0):
            # New chunk. Activate it
            chunk_key = self.n_active_chunks
            self.active_chunks = np.append(self.active_chunks, chunk_value)
            
            # Allocate and set new chunk pixels to 0
            self.cmap = np.append(self.cmap, np.zeros(self.npix_per_chunk, dtype=self.data_type_cmap) )
            self.n_active_chunks +=1
            
            #Check active_chunks does not have repetitions
            max_rep = np.unique(self.active_chunks, return_counts=True)[1].max()
            if(max_rep>1):
                raise Exception(f"Error activating chunk {chunk_value}" +
                                f"max_rep: {max_rep}" +
                                f"active_chunks: {self.active_chunks}")
            
        if((n is not 0) and (n is not 1)):
            print("Error determining chunk position " + str(chunk_value)
                  + ", found in " + str(n) + " active_chunks: " + str(chunk_key))
        
        # Add count
        ipix_chunk = ipix_full % self.npix_per_chunk + chunk_key*self.npix_per_chunk
        self.cmap[ipix_chunk] += count_increase

        # Check count conservation
        if(int(self.cmap.sum()) != int(n_old+count_increase)):
            raise Exception( (f"Error adding count, ipix_full={ipix_full}, "
                                f"ipix_chunk={ipix_chunk}, {self.cmap.sum()}, {n_old}") )
        
        
    def full_map2chunk_map(self, map_full):
        """
        Transform a full-size map to a chunk map.
        """
        # Check size
        if(map_full.size != self.npix):
            raise Exception(f"Error: map size needs to have {self.npix} pixels (nside: {self.nside})")
        
        for i, m in enumerate(map_full):
            if(m>0):
                self.increase_ipix_count(i, count_increase=m)
        
        # Check count conservation
        if(int(self.cmap.sum()) != map_full.sum()):
            raise Exception( (f"Error creating the chunk map, "
                              f"{int(self.cmap.sum())}, {map_full.sum()}") )
        
        
    def reconstruct_full_map(self):
        """
        Reconstruct and return a full-size map from the chunk map.
        """
       
        # Sanity check
        if(self.active_chunks.size != self.n_active_chunks):
            raise Exception( (f"Error handling active chunks, "
                              f"{self.active_chunks} size is {self.n_active_chunks}") )
        
        map_full = np.zeros(self.npix, dtype=self.data_type_cmap)
        
        # Fill in values in the full map for the active chunks
        for k in range(self.n_active_chunks):
            i0, i1 = self.chunk2ipix_ranges(self.active_chunks[k])
            j0, j1 = self.chunk2ipix_ranges(k)
            map_full[i0:i1+1] = self.cmap[j0:j1+1]
            
        # Check count conservation
        if(int(self.cmap.sum()) != map_full.sum()):
            raise Exception( (f"Error reconstructing full map, "
                              f"{int(self.cmap.sum())}, {map_full.sum()}") )
        
        return map_full
    
    
    def create_map_active_chunks(self):
        """
        Get a full-size map with 1s where a chunk is active and 0s otherwise.
        """
        
        cmap = Chunkpix(self.nside, self.nside_chunks)
        # Set ones in the chunk map
        cmap.cmap = np.ones_like(self.cmap)
        cmap.active_chunks = self.active_chunks
        cmap.n_active_chunks = self.n_active_chunks
        # Reconstruct it
        return cmap.reconstruct_full_map()
        
        
    def create_map_chunk_numbers(self):
        """
        Return a full-size map that contains chunk numbers
        """
        return self.ipix2chunk(np.arange(self.npix))


