from typing import Protocol, runtime_checkable, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# Parameter Configurations
# =============================================================================

class MPCleanerConfig(BaseModel):
    """
    Configuration for Marchenko-Pastur cleaner.
    
    Pydantic handles validation, type coercion, serialization.
    """
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    threshold_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Multiplier for MP upper edge (lambda_+)"
    )
    
    @field_validator('threshold_multiplier')
    @classmethod
    def validate_reasonable_range(cls, v: float) -> float:
        if v < 0.5:
            logger.warning(
                f"threshold_multiplier={v} is very aggressive. "
                "Consider values closer to 1.0 for stable results."
            )
        return v


class RIECleanerConfig(BaseModel):
    """Configuration for Rotationally Invariant Estimator."""
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    grid_size: Optional[int] = Field(
        default=None,
        gt=10,
        le=5000,
        description="KDE/Hilbert grid size. None = adaptive"
    )
    bandwidth: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="KDE bandwidth. None = Silverman's rule"
    )
    
    @field_validator('grid_size')
    @classmethod
    def validate_grid_size(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v > 2000:
            logger.warning(
                f"grid_size={v} may be slow. Consider reducing to <1000 "
                "unless high precision is critical."
            )
        return v

# =============================================================================
# Protocol Interface
# =============================================================================

@runtime_checkable
class CovarianceCleaner(Protocol):
    """
    Protocol for covariance matrix noise cleaning.    
    Any class implementing these methods automatically satisfies this protocol.
    """
    
    def fit(
        self, 
        C_emp: np.ndarray, 
        data_shape: Optional[Tuple[int, int]] = None
    ) -> 'CovarianceCleaner':
        """
        Fit the cleaner to an empirical covariance matrix.
        
        Parameters
        ----------
        C_emp : np.ndarray
            Empirical covariance/correlation matrix.
        data_shape : tuple of (int, int), optional
            (n_samples, n_features) from original data.
        
        Returns
        -------
        self
            Fitted cleaner instance.
        """
        ...
    
    def get_cleaned_covariance(self) -> np.ndarray:
        """
        Return the cleaned covariance matrix.
        
        Returns
        -------
        np.ndarray
            Cleaned covariance matrix.
        """
        ...

# =============================================================================
# Cleaner Implementations
# =============================================================================

class IdentityCleaner:
    """
    A fallback class for when cleaning is disabled.
    """
    def fit(
        self, 
        C_emp: np.ndarray, 
        data_shape=None
    ):
        self.C_cleaned_ = C_emp.copy()
        return self
    def get_cleaned_covariance(self):
        return self.C_cleaned_.copy()

class MarchenkoPasturCleaner:
    """
    Marchenko-Pastur threshold-based cleaning.
    """
    
    def __init__(
        self, 
        config: Optional[MPCleanerConfig] = None,
        **kwargs
    ):
        """
        Initialize cleaner.
        
        Parameters
        ----------
        config : MPCleanerConfig, optional
            Pre-configured settings. If None, create from kwargs.
        **kwargs
            Parameters passed to MPCleanerConfig if config is None.
        """
        self.config = config or MPCleanerConfig(**kwargs)
        
        # Fitted state
        self.C_cleaned_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.lambda_plus_ = None
    
    def fit(
        self, 
        C_emp: np.ndarray, 
        data_shape: Optional[Tuple[int, int]] = None,
        verbose: bool = False
    ) -> 'MarchenkoPasturCleaner':
        """Fit the Marchenko-Pastur cleaner."""
        self._validate_matrix(C_emp)
        
        if data_shape is None:
            raise ValueError("data_shape required for Marchenko-Pastur cleaning")
        
        n_samples, n_features = data_shape
        q = n_features / n_samples
        self.lambda_plus_ = self.config.threshold_multiplier * (1 + np.sqrt(q)) ** 2
        
        eigenvalues, eigenvectors = np.linalg.eigh(C_emp)
        cleaned_eigenvalues = np.where(eigenvalues >= self.lambda_plus_, eigenvalues, 0.0)
        
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.C_cleaned_ = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T
        
        n_signal = int(np.sum(eigenvalues >= self.lambda_plus_))
        if verbose:
            logger.info(
                f"MP cleaning: lambda_+ = {self.lambda_plus_:.4f}, "
                f"kept {n_signal}/{n_features} eigenvalues"
            )
        
        return self
    
    def get_cleaned_covariance(self) -> np.ndarray:
        """Return cleaned covariance matrix."""
        if self.C_cleaned_ is None:
            raise ValueError("Cleaner not fitted. Call fit() first.")
        return self.C_cleaned_.copy()
    
    def _validate_matrix(self, C: np.ndarray) -> None:
        """Validate matrix is square and symmetric."""
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {C.shape}")
        if not np.allclose(C, C.T, atol=1e-8):
            raise ValueError("Matrix must be symmetric")

class RotationallyInvariantEstimator:
    f"""
    Rotationally invariant estimator (RIE) for free multiplicative noise models of the form: 

        M = sqrt(C) O B O^dag sqrt(C),

        where 
            - M is the noisy measurement matrix
            - C is the signal matrix, required to be a positive definite
            - O is a random matrix chosen from the Orthogonal group O(N) according to Haar measure
        
    The optimal rotationally invariant estimator is given by
        hat(Xi) = lambda gamma_B(lambda) +  (lambda h_M(lambda) - 1) omega_B(lambda)
    
    Implements estimator from the paper [1].

    References
    ----------
        [1] Bun, J., et al. "Rotational invariant estimator for general noisy matrices." arXiv:1502.06736 (2015).
    """
    
    def __init__(
        self,
        config: Optional[RIECleanerConfig] = None,
        **kwargs
    ):
        """
        Initalise the RIE cleaner.

        Parameters
        ----------
        grid_size : int, optional
            Number of points to be used in the KDE/Hilbert grid. Higher values improve accuracy but increase compute time O(grid_size^2).
        """
        self.config = config or RIECleanerConfig(**kwargs)
        self.bandwidth = self.config.bandwidth
        self.grid_size = self.config.grid_size
        
        # Fitted state
        self.q = None
        self.eigenvalues_M_ = None
        self.eigenvectors_ = None
        self.cleaned_eigenvalues_ = None
        self.C_cleaned_ = None
        self.x_grid_ = None
        self.rho_M_ = None
        self.h_M_grid_ = None
        self.grid = None
            
        
    def fit(
        self, 
        M: np.ndarray, 
        data_shape: Optional[Tuple[int, int]] = None,
        return_eigenvectors: bool = True,
        verbose: bool = False
    ) -> 'RotationallyInvariantEstimator':
        """
        Fit the estimator to the noisy matrix M.
        
        Computes the eigendecomposition, estimates the spectral density via KDE, precomputes the Hilbert transform,
        and applies the RIE cleaning to obtain shrunk eigenvalues.

        Parameters
        ----------
        M : array-like, shape (n, n)
            Noisy sample covariance matrix (must be symmetric and positive semi-definite).
        data_shape: int tuple
            Shape of data M was computed from. 
        return_eigenvectors : bool, optional
            If True, compute and store eigenvectors for later reconstruction (default: True).
            
        Returns
        -------
        self : BunMultiplicativeNoiseEstimator
            Fitted estimator.
        """
        
        self._validate_matrix(M)
        
        if data_shape is None:
            raise ValueError("data_shape required for RIE")
        
        n_samples, n_features = data_shape
        self.q = n_features / n_samples

        M = np.asarray(M)
        n = M.shape[0]
        
        if self.grid_size is None:
            # Use 3-5x the matrix size, capped at 1000
            self.grid_size = min(max(3 * n, 300), 1000)
        
        # Compute eigendecomposition
        if return_eigenvectors:
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            self.eigenvectors_ = eigenvectors
        else:
            eigenvalues = np.linalg.eigvalsh(M)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        if return_eigenvectors:
            self.eigenvectors_ = self.eigenvectors_[:, idx]
        
        self.eigenvalues_M_ = eigenvalues
        
        # Compute density of M
        self.x_grid_, self.rho_M_ = self.compute_density_kde(eigenvalues)
        
        # Pre-compute Hilbert transform ONCE on the entire grid
        self.h_M_grid_ = self.compute_hilbert_grid(self.x_grid_, self.rho_M_)
        
        # Apply cleaning formula
        self.cleaned_eigenvalues_ = self._clean_eigenvalues(eigenvalues)
        
        if verbose:
            logger.info(
                f"RIE cleaning: q={self.q_:.4f}, grid_size={grid_size}, "
                f"eigenvalue range: [{self.cleaned_eigenvalues_.min():.4f}, "
                f"{self.cleaned_eigenvalues_.max():.4f}]"
            )
        
        return self

    
    def _clean_eigenvalues(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        Apply RIE cleaning formula to the observed eigenvalues.
        
        Interpolates precomputed spectral transforms and applies vectorized shrinkage.

        Parameters
        ----------
        eigenvalues : ndarray
            Observed eigenvalues of M.
        """

        # Fast linear interpolation from precomputed grids
        h_M = np.interp(eigenvalues, self.x_grid_, self.h_M_grid_)
        rho_M = np.interp(eigenvalues, self.x_grid_, self.rho_M_) # observed spectral density
        
        # Stieltjes transform on the real axis
        g_M = h_M + 1j * np.pi * rho_M
        
        # Denominator (complex modulus squared)
        denom = np.abs(1 - self.q + self.q * eigenvalues * g_M) ** 2
        denom = np.maximum(denom, 1e-12)   # numerical stability
        
        gamma_B = (1 - self.q + self.q * eigenvalues * h_M) / denom
        omega_B = -self.q * eigenvalues / denom
        
        # Compute the cleaned eigenvalues
        cleaned = eigenvalues * gamma_B + (eigenvalues * h_M - 1.0) * omega_B
        
        # Safeguard: tiny numerical negatives are impossible for a covariance
        return np.maximum(cleaned, 1e-8)
    

    def get_cleaned_covariance(self) -> np.ndarray:
        """
        Construct cleaned covariance matrix using cleaned eigenvalues and original eigenvector basis.
        """
        if self.eigenvectors_ is None:
            raise ValueError("Must fit with return_eigenvectors=True")
        
        return self.eigenvectors_ @ np.diag(self.cleaned_eigenvalues_) @ self.eigenvectors_.T

    def compute_density_kde(
        self,
        eigenvalues: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the spectral density of eigenvalues using Gaussian KDE.

        Uses Silverman's rule for bandwidth and a uniform grid for efficient Hilbert computation.

        Parameters
        ----------
        bandwidth : float, optional
            KDE bandwidth (default: Silverman's rule).
        
        Returns
        -------
        x : ndarray
            Grid points.
        rho : ndarray
            Estimated density at grid points.
        """
        eigenvalues = np.asarray(eigenvalues).ravel() # Shape: (n,)
        n = len(eigenvalues)
        
        if self.bandwidth is None:
            # Silverman's rule of thumb
            bandwidth = 1.06 * np.std(eigenvalues) * n ** (-0.2)
            self.bandwidth = max(bandwidth, 1e-6 * (eigenvalues.max() - eigenvalues.min() + 1e-10)) # to devision by zero for identical eigenvalues 
        
        # Create uniform grid
        margin = 3 * self.bandwidth
        x_min, x_max = eigenvalues.min() - margin, eigenvalues.max() + margin
        x = np.linspace(x_min, x_max, self.grid_size) # grid_size points
        
        # Vectorized KDE with broadcasting, use Gaussian kernel.
        diffs = x[:, None] - eigenvalues[None, :] # Shape: (grid_size, n)
        
        # Evaluate Gaussian kernel on the grid
        kernels = np.exp(-0.5 * (diffs / self.bandwidth) ** 2) # Shape: (grid_size, n)
        
        # Sum over eigenvalues for each point
        rho = kernels.sum(axis=1) # Shape: (grid_size,)

        # Normalise
        rho /= (n * self.bandwidth * np.sqrt(2 * np.pi))
        
        return x, rho


    def compute_hilbert_grid(
        self,
        x: np.ndarray, 
        rho: np.ndarray
    ) -> np.ndarray:
        """
        Compute Hilbert transform
            h(lambda) = PV int rho(mu)/(lmabda - mu) dmu
        
        Use vectorised broadcasting and masked principal value computation.
        
        Parameters
        ----------
        x : ndarray
            Grid points.
        rho : ndarray
            Density values.

        Returns
        -------
        h : ndarray
            Hilbert transform values.
        """
        dx = x[1] - x[0]
        
        # Create a matrix of differences by broadcasting
        # X[i,j] = x[i] - x[j]
        X = x[:, None] - x[None, :] # Shape: (grid_size, grid_size)
        
        # Mask excluding singularity neighbourhood to approximate Cauchy PV
        mask = np.abs(X) > 3 * dx
        
        # Safe vectorised division where mask is True
        integrand = np.zeros_like(X, dtype=float)
        np.divide(rho[None, :], X, out=integrand, where=mask)
        
        h = np.sum(integrand, axis=1) * dx
        return h
        
    def _validate_matrix(self, M: np.ndarray) -> None:
        """Validate matrix input"""
        if not np.allclose(M, M.T, atol=1e-8):
            raise ValueError("Input matrix M must be symmetric.")
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Input matrix M must be square, got: {M.shape}.")

# =============================================================================
# Covariance Cleaner Factory
# =============================================================================

class CleanerType(str, Enum):
    MP = "marchenko_pastur"
    RIE = "rie"
    IDENTITY = "identity"


def create_covariance_cleaner(
    cleaner_type: CleanerType,
    **kwargs
) -> CovarianceCleaner:
    """Factory method for a covariance cleaner class."""

    if cleaner_type == CleanerType.IDENTITY:
        return IdentityCleaner()

    elif cleaner_type == CleanerType.MP:
        return MarchenkoPasturCleaner(**kwargs)

    elif cleaner_type == CleanerType.RIE:
        return RotationallyInvariantEstimator(**kwargs)

    else:
        raise ValueError(f"Unsupported CleanerType: {cleaner_type}.")
