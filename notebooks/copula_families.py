# src/copula_families.py
"""
Copula Families Implementation for Visualization Project
Includes Gaussian, Student-t, and Clayton copulas with BIC calculation
"""

import numpy as np
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============================================
# BASE COPULA CLASS
# ============================================

class BaseCopula:
    """Base class for all copula families"""
    
    def __init__(self):
        self.params = {}
        self.n_params = 0
        self.name = "Base"
        
    def fit(self, u, v):
        """Estimate parameters from uniform marginals"""
        raise NotImplementedError("Subclasses must implement fit()")
        
    def pdf(self, u, v):
        """Probability density function"""
        raise NotImplementedError("Subclasses must implement pdf()")
    
    def log_likelihood(self, u, v):
        """Log-likelihood of the data"""
        return np.sum(np.log(self.pdf(u, v) + 1e-10))
    
    def bic(self, u, v):
        """Bayesian Information Criterion (lower is better)"""
        n = len(u)
        return -2 * self.log_likelihood(u, v) + self.n_params * np.log(n)
    
    def aic(self, u, v):
        """Akaike Information Criterion (lower is better)"""
        return -2 * self.log_likelihood(u, v) + 2 * self.n_params
    
    def __repr__(self):
        params_str = ", ".join([f"{k}={v:.4f}" for k, v in self.params.items()])
        return f"{self.name}Copula({params_str})"

# ============================================
# GAUSSIAN COPULA
# ============================================

class GaussianCopula(BaseCopula):
    """Gaussian (Normal) copula with linear dependence"""
    
    def __init__(self):
        super().__init__()
        self.n_params = 1  # ρ (correlation)
        self.name = "Gaussian"
        
    def fit(self, u, v):
        """
        Fit Gaussian copula using method of moments
        
        Parameters:
        -----------
        u, v : array-like
            Uniform marginals in [0,1]
        """
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        
        # Transform to normal quantiles
        x = norm.ppf(u)
        y = norm.ppf(v)
        
        # Estimate correlation
        self.params['rho'] = np.corrcoef(x, y)[0, 1]
        self.params['rho'] = np.clip(self.params['rho'], -0.99, 0.99)
        
        return self
        
    def pdf(self, u, v):
        """Gaussian copula density - CORRECTED"""
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        
        x = norm.ppf(u)
        y = norm.ppf(v)
        rho = self.params['rho']
        
        if abs(rho) >= 1:
            return np.ones_like(u)
        
        det = 1 - rho**2
        
        # Correct Gaussian copula density formula
        exponent = -(x**2 - 2*rho*x*y + y**2) / (2*det) + (x**2 + y**2) / 2
        return (1 / np.sqrt(det)) * np.exp(exponent)

# ============================================
# STUDENT-T COPULA  
# ============================================

class StudentTCopula(BaseCopula):
    """Student-t copula with tail dependence"""
    
    def __init__(self):
        super().__init__()
        self.n_params = 2  # ρ (correlation) and ν (degrees of freedom)
        self.name = "Student-t"
        
    def fit(self, u, v, nu_range=(2, 30)):
        """
        Fit Student-t copula using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        u, v : array-like
            Uniform marginals in [0,1]
        nu_range : tuple
            Range for degrees of freedom ν search
        """
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        
        # Initial correlation estimate using Gaussian approximation
        x_norm = norm.ppf(u)
        y_norm = norm.ppf(v)
        rho_initial = np.corrcoef(x_norm, y_norm)[0, 1]
        rho_initial = np.clip(rho_initial, -0.99, 0.99)
        
        # Negative log-likelihood function
        def neg_log_lik(params):
            rho, nu = params[0], params[1]
            
            # Parameter constraints
            if abs(rho) >= 0.99 or nu < nu_range[0] or nu > nu_range[1]:
                return 1e10
                
            # Transform to t-distribution quantiles
            x_t = student_t.ppf(u, df=nu)
            y_t = student_t.ppf(v, df=nu)
            
            # Bivariate t-density
            rho = np.clip(rho, -0.99, 0.99)
            det = 1 - rho**2
            quad = (x_t**2 + y_t**2 - 2 * rho * x_t * y_t) / det
            
            # Log-density of bivariate t
            log_density = (-np.log(2*np.pi) - 0.5*np.log(det) - 
                          ((nu+2)/2)*np.log(1 + quad/nu) + 
                          ((nu+1)/2)*np.log(1 + x_t**2/nu) + 
                          ((nu+1)/2)*np.log(1 + y_t**2/nu))
            
            return -np.sum(log_density)
        
        # Optimize using bounded optimization
        result = minimize(neg_log_lik, [rho_initial, 5], 
                         bounds=[(-0.99, 0.99), nu_range],
                         method='L-BFGS-B')
        
        self.params['rho'] = result.x[0]
        self.params['nu'] = result.x[1]
        
        return self
        
    def pdf(self, u, v):
        """Student-t copula density - CORRECTED"""
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        
        rho = self.params['rho']
        nu = self.params['nu']
        
        # Transform to t-distribution quantiles
        x = student_t.ppf(u, df=nu)
        y = student_t.ppf(v, df=nu)
        
        if abs(rho) >= 0.99:
            return np.ones_like(u)
        
        det = 1 - rho**2
        quad = (x**2 + y**2 - 2*rho*x*y) / det
        
        # Bivariate t density
        bivariate_dens = (1 / (2*np.pi*np.sqrt(det)) * 
                        (1 + quad/nu) ** (-(nu+2)/2))
        
        # Univariate t densities
        univariate_x = student_t.pdf(x, df=nu)
        univariate_y = student_t.pdf(y, df=nu)
        
        # Copula density
        density = bivariate_dens / (univariate_x * univariate_y)
        
        return np.clip(density, 1e-10, 1e10)

# ============================================
# CLAYTON COPULA
# ============================================

class ClaytonCopula(BaseCopula):
    """Clayton copula with lower tail dependence"""
    
    def __init__(self):
        super().__init__()
        self.n_params = 1  # θ (dependence parameter)
        self.name = "Clayton"
        
    def fit(self, u, v):
        """
        Fit Clayton copula using Kendall's tau method
        
        Parameters:
        -----------
        u, v : array-like
            Uniform marginals in [0,1]
        """
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        
        # Calculate empirical Kendall's tau
        n = len(u)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i+1, n):
                prod = (u[i] - u[j]) * (v[i] - v[j])
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1
        
        # Kendall's tau = (concordant - discordant) / (concordant + discordant)
        total = concordant + discordant
        if total == 0:
            tau = 0
        else:
            tau = (concordant - discordant) / total
        
        # Clayton only models positive dependence
        tau = np.clip(tau, 0.01, 0.99)
        
        # Relationship: τ = θ/(θ+2) → θ = 2τ/(1-τ)
        self.params['theta'] = 2 * tau / (1 - tau)
        self.params['theta'] = np.clip(self.params['theta'], 0.01, 10)
        
        return self
        
    def pdf(self, u, v):
        """
        Clayton copula density: c(u,v) = (1+θ)(uv)^(-θ-1)(u^(-θ)+v^(-θ)-1)^(-2-1/θ)
        """
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        
        theta = self.params['theta']
        
        # Clayton copula density formula
        return ((1 + theta) * (u * v)**(-theta - 1) * 
                (u**(-theta) + v**(-theta) - 1)**(-2 - 1/theta))

# ============================================
# HELPER FUNCTIONS
# ============================================

def transform_to_uniform(data, marginal_type='gaussian'):
    """
    Transform data to uniform marginals [0,1] via CDF
    
    Parameters:
    -----------
    data : array-like or tuple of arrays
        Data to transform. If tuple, transform each array separately.
    marginal_type : str
        'gaussian' : Assume Gaussian marginals
        'empirical' : Use empirical CDF (not implemented)
        
    Returns:
    --------
    u : array or tuple of arrays
        Uniform marginals in [0,1]
    """
    if isinstance(data, tuple):
        # Multiple arrays
        return tuple(transform_to_uniform(d, marginal_type) for d in data)
    
    data = np.asarray(data)
    
    if marginal_type == 'gaussian':
        mean = np.mean(data)
        std = np.std(data)
        u = norm.cdf(data, loc=mean, scale=std)
    else:
        # Empirical CDF (simplified)
        sorted_data = np.sort(data)
        u = np.searchsorted(sorted_data, data) / len(data)
    
    # Clip to avoid 0 or 1
    u = np.clip(u, 1e-10, 1-1e-10)
    return u

def compare_copulas(u, v, copula_classes=None):
    """
    Compare multiple copulas using BIC
    
    Parameters:
    -----------
    u, v : array-like
        Uniform marginals
    copula_classes : list, optional
        List of copula classes to compare
        
    Returns:
    --------
    results : dict
        Dictionary with BIC scores and fitted copulas
    """
    if copula_classes is None:
        copula_classes = [GaussianCopula, StudentTCopula, ClaytonCopula]
    
    results = {}
    
    for CopulaClass in copula_classes:
        try:
            copula = CopulaClass()
            copula.fit(u.copy(), v.copy())
            bic = copula.bic(u, v)
            
            results[CopulaClass.__name__] = {
                'copula': copula,
                'bic': bic,
                'log_likelihood': copula.log_likelihood(u, v),
                'params': copula.params.copy()
            }
        except Exception as e:
            print(f"Error fitting {CopulaClass.__name__}: {e}")
            results[CopulaClass.__name__] = {
                'copula': None,
                'bic': np.inf,
                'error': str(e)
            }
    
    return results