import numpy as np
import scipy.linalg as lg
import xarray as xr

"""
A Class to calculate eofs and pcs of a given dataset
Requires geo-spatial data
"""
class Eofs():
    def __init__(self, data: xr.DataArray, cosweight: bool=True):
        """
        Docstring for __init__
        
        :param self: Description
        :param data: Description
        :type data: xr.DataArray
        :param weights: Description
        :type weights: xr.DataArray
        """
        # Data to calculate eofs and pcs
        # Should be an xarray DataArray with lat, lon and time as coordinates
        # TODO: Throw an error if data does not have lat, lon and time coords
        # If cosweight, cosine weight data
        if cosweight:
            self._data = data*np.sqrt(np.cos(data.lat * np.pi/180))
        # Otherwise, do nothing
        else:
            self._data = data

        # Do svd
        self._eofs, self._evals, self._pcs = self.__svd__()
        
    # A private method to do singular value decomposition and return the eofs, eigenvalues, and pcs
    def __svd__(self):
        # Getting original shape
        shape = self._data.transpose('lat', 'lon', 'time').shape
        # Stacking dimensions and transposing so U is the spatial covariance eigenvectors
        # and V is the temporal covariance eigenvectors
        ds = self._data.stack(space=('lat', 'lon')).transpose('space', 'time')
        # Filling any nans
        ds = ds.fillna(0)
        # Doing svd
        eofs, sigma, pcs = lg.svd(ds)
        # Grabbing number of eofs
        neofs=len(sigma)

        # ---------------------------------------------------------- #
        # Transposing eofs so rows are the vectors, not columns
        eofs = eofs.transpose()
        # Reshaping eofs
        eofs = eofs[:neofs].reshape(neofs, shape[0], shape[1])
        # Taking sqrt of sigma to get eigenvalues of covariance matrix
        evals = sigma ** 2 / (len(self._data.time) - 1)

        # Constructing pcs
        # NOTE: We are constructing a diagonal matrix out of the eigenvalues and then
        # NOTE: multiplying V by it. Is there a more efficient way to do this? With some
        # NOTE: sparce matrix math or something? Maybe not worth it
        # First, make diagonal matrix with correct dimensions out of eigenvalues
        diags = np.zeros((neofs, len(self._data.time)))
        np.fill_diagonal(diags, sigma)
        pcs = diags @ pcs

        # EOF index
        index = np.arange(0, neofs)

        # Putting eofs into xarray DataArray
        eofs = xr.DataArray(
            data=eofs,
            dims=('eof', 'lat', 'lon'),
            coords={
                'eof': index,
                'lat': self._data.lat.values,
                'lon': self._data.lon.values
            }
        )
        
        # Applying a land mask
        eofs = xr.where(np.isnan(self._data.isel(time=0)), np.nan, eofs)

        # Putting pcs into xarray DataArray
        pcs = xr.DataArray(
            data=pcs,
            dims=('pc', 'time'),
            coords={
                'pc': index,
                'time': self._data.time.values
            }
        )

        # Putting eigenvalues into xarray DataArray
        evals = xr.DataArray(
            data=evals,
            dims=('eof'),
            coords={
                'eof': index
            }
        )

        # pcs = pcs / (np.sqrt(evals)).rename({'eof': 'pc'})
        
        return (eofs, evals, pcs)

    def getEofs(self, neofs: int=None):
        """
        Gives empirical orthogonal functions (eofs) of dataset.
        
        Parameters
        ----------
        neofs(int): Number of eofs to return. If none is given, all eofs are returned.

        Returns
        -------
        xarray DataArray of eofs
        """
        if neofs==None:
            return self._eofs
        # TODO: Throw an error if neofs is larger than the number of eofs
        # This should probably just be a method, since we need to use it multiple times
        else:
            return self._eofs.sel(eof=slice(0, neofs-1))
        
    def getPcs(self, pcscaling: int=0, npcs: int=None):
        """
        Gives principal components (pcs) of dataset
        
        Parameters
        ----------
         pcscaling(int): Kind of scaling for pcs
        *0* returns pcs without any scaling applied (default)

        *1* returns normalized pcs (i.e. divided by the sqrt of eigenvalues)

        npcs(int) Number of pcs to return, by default returns all

        Returns
        -------
        xarray DataArray of pcs, scaled as requested
        """
        # If npcs not specified, return all of them
        # TODO: Throw an error if npcs is larger than the number of pcs
        if npcs == None:
            npcs = self.numEofs()
        # If pcscaling is 1, return normalized pcs (divided by the sqrt of eigenvalues)
        if pcscaling == 1:
            return (self._pcs / np.sqrt(self._evals).rename({'eof': 'pc'})).isel(pc=slice(0, npcs-1))
        # Otherwise, if pcscaling is 0, return pcs as is
        return self._pcs.isel(pc=slice(0, npcs-1))

    def getEigenvalues(self, neofs: int=None):
        if neofs==None:
            return self._evals
        # TODO: Throw an error if neofs is larger than the number of eofs
        else:
            return self._evals.sel(eof=slice(0, neofs-1))
        
    def numEofs(self):
        return len(self._eofs.eof)