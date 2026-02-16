import time
import numpy as np
from scipy.stats import norm, kendalltau, spearmanr, skew, kurtosis
import sys
sys.path.append('../src')
from copula_families import GaussianCopula, StudentTCopula, ClaytonCopula

def generate_random_features(n=1000):
    """Generate two random correlated features"""
    rho = np.random.uniform(-0.9, 0.9)
    x = np.random.normal(0, 1, n)
    y = rho * x + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)
    
    # Transform to uniform via CDF
    u = norm.cdf(x)
    v = norm.cdf(y)
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    
    return u, v, rho

def calculate_all_features(u, v, rho, q=0.1):
    """Calculate all proposed features for classifier"""
    features = {}
    
    # Start timing
    start_time = time.time()
    
    # 1. Tail dependence coefficients
    features['lambda_l'] = np.mean((u <= q) & (v <= q)) / q
    features['lambda_r'] = np.mean((u > 1-q) & (v > 1-q)) / q
    
    # 2. Linear correlation (already have rho from generation)
    features['rho'] = rho
    
    # 3. Derived tail features
    features['tail_asymmetry'] = features['lambda_l'] - features['lambda_r']
    features['tail_strength'] = (features['lambda_l'] + features['lambda_r']) / 2
    features['tail_ratio'] = (min(features['lambda_l'], features['lambda_r']) / 
                             max(features['lambda_l'], features['lambda_r']) if 
                             max(features['lambda_l'], features['lambda_r']) > 0 else 0)
    features['lambda_l_norm'] = features['lambda_l'] / (features['rho'] + 0.01)
    features['lambda_r_norm'] = features['lambda_r'] / (features['rho'] + 0.01)
    
    # 4. Rank correlations
    tau, _ = kendalltau(u, v)
    features['kendalls_tau'] = tau if not np.isnan(tau) else 0
    
    spearman, _ = spearmanr(u, v)
    features['spearman_rho'] = spearman if not np.isnan(spearman) else 0
    
    # 5. Gaussian deviation measure
    features['tau_rho_diff'] = abs(features['kendalls_tau'] - (2/np.pi)*np.arcsin(features['rho']))
    
    # 6. Marginal moments
    features['skew_u'] = skew(u)
    features['skew_v'] = skew(v)
    features['kurt_u'] = kurtosis(u)
    features['kurt_v'] = kurtosis(v)
    
    # End timing
    features['time_features'] = time.time() - start_time
    
    return features

def fit_all_copulas(u, v):
    """Fit all three copulas and return AICs"""
    times = {}
    aics = {}
    
    # Gaussian
    start = time.time()
    gaussian = GaussianCopula()
    gaussian.fit(u, v)
    times['Gaussian'] = time.time() - start
    aics['Gaussian'] = gaussian.aic(u, v)
    
    # Clayton
    start = time.time()
    clayton = ClaytonCopula()
    clayton.fit(u, v)
    times['Clayton'] = time.time() - start
    aics['Clayton'] = clayton.aic(u, v)
    
    # Student-t
    start = time.time()
    student_t = StudentTCopula()
    student_t.fit(u, v)
    times['Student_t'] = time.time() - start
    aics['Student_t'] = student_t.aic(u, v)
    
    return times, aics

def main():
    # Configuration
    n_samples = 1000  # Sample size
    n_trials = 100    # Number of trials
    
    # Storage for results
    feature_times = []
    copula_times = []
    best_copulas = []
    
    print("TIMING COMPARISON: Feature Calculation vs Copula Fitting")
    print("="*60)
    print(f"Running {n_trials} trials with n={n_samples} samples each")
    print()
    
    for trial in range(n_trials):
        # Generate random data
        u, v, rho = generate_random_features(n_samples)
        
        # Time feature calculation
        features = calculate_all_features(u, v, rho)
        feature_times.append(features['time_features'])
        
        # Time copula fitting
        times, aics = fit_all_copulas(u, v)
        copula_times.append(sum(times.values()))
        
        # Determine best copula
        best_copula = min(aics, key=aics.get)
        best_copulas.append(best_copula)
        
        # Progress
        if (trial + 1) % 20 == 0:
            print(f"Completed {trial + 1}/{n_trials} trials")
    
    # Calculate statistics
    avg_feature_time = np.mean(feature_times)
    avg_copula_time = np.mean(copula_times)
    std_feature_time = np.std(feature_times)
    std_copula_time = np.std(copula_times)
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nAverage times per dataset (n={n_samples}):")
    print(f"  Feature calculation: {avg_feature_time*1000:.2f} ± {std_feature_time*1000:.2f} ms")
    print(f"  Copula fitting (all 3): {avg_copula_time*1000:.2f} ± {std_copula_time*1000:.2f} ms")
    
    speedup = avg_copula_time / avg_feature_time
    print(f"\nSpeedup factor: {speedup:.2f}x (features are {speedup:.1f}x faster)")
    
    print(f"\nCopula distribution (best by AIC):")
    for copula in ['Gaussian', 'Clayton', 'Student_t']:
        count = best_copulas.count(copula)
        percentage = 100 * count / n_trials
        print(f"  {copula}: {count} ({percentage:.1f}%)")
    
    # FIXED: Detailed breakdown - extract just the u,v from generate_random_features
    print(f"\nDetailed timing breakdown (ms):")
    print(f"  Feature calculation: {avg_feature_time*1000:.2f} ms")
    
    # Calculate average fitting times separately
    gaussian_times = []
    clayton_times = []
    student_times = []
    
    for _ in range(20):
        u, v, _ = generate_random_features(n_samples)  # Only use u,v, ignore rho
        times, _ = fit_all_copulas(u, v)
        gaussian_times.append(times['Gaussian'])
        clayton_times.append(times['Clayton'])
        student_times.append(times['Student_t'])
    
    print(f"  Gaussian fitting: {np.mean(gaussian_times)*1000:.2f} ms")
    print(f"  Clayton fitting: {np.mean(clayton_times)*1000:.2f} ms")
    print(f"  Student-t fitting: {np.mean(student_times)*1000:.2f} ms")
    
    # Alternative one-liner fix (also works):
    # print(f"  Gaussian fitting: {np.mean([fit_all_copulas(*generate_random_features(n_samples)[:2])[0]['Gaussian'] for _ in range(20)])*1000:.2f} ms")
    
    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if speedup > 2:
        print(f"✓ Use feature-based classifier!")
        print(f"  Features are {speedup:.1f}x faster than fitting all copulas")
        print(f"  Saves {(avg_copula_time - avg_feature_time)*1000:.1f} ms per dataset")
    else:
        print(f"✓ Just fit all copulas")
        print(f"  Speed difference is minimal ({speedup:.1f}x)")
        print(f"  Simpler implementation, no classifier needed")
    
    # Show example features
    print("\nExample features calculated:")
    u, v, rho = generate_random_features(100)
    example_features = calculate_all_features(u, v, rho)
    for key, value in list(example_features.items())[:8]:  # First 8 features
        if key != 'time_features':
            print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()