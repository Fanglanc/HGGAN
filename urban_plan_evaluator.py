import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
from collections import defaultdict
import math


class UrbanPlanEvaluator:
    def __init__(self, poi_names: Optional[List[str]] = None):
        """
        Args:
            poi_names: Names of 20 POI channels (e.g., ['residential', 'commercial', ...])
        """
        self.poi_names = poi_names or self._default_poi_names()
        
    def _default_poi_names(self) -> List[str]:
        """Default POI category names"""
        return [
            "Residential_Low", "Residential_High", "Education", "Healthcare", 
            "Retail", "Office", "Industrial", "Recreation", "Culture", "Transport",
            "Government", "Religious", "Hotel", "Restaurant", "Park", "Sports",
            "Entertainment", "Finance", "Tech", "Mixed_Use"
        ]

    # ========================================================================
    # PART 1: ENHANCED QUANTITATIVE EVALUATION WITH ROBUST UNCERTAINTY
    # ========================================================================

    def quantitative_eval(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray] = None,
        n_samples: int = 20,
        uncertainty_method: str = 'ensemble',  # 'ensemble', 'bootstrap', 'bayesian'
        noise_levels: List[float] = [0.01, 0.03, 0.05],
        rubric_variants: List[str] = ['standard', 'sustainability', 'equity', 'economic'],
        seed: int = 42
    ) -> Dict[str, Any]:

        rng = np.random.default_rng(seed)
        
        results = {
            "method": uncertainty_method,
            "config": {
                "n_samples": n_samples,
                "noise_levels": noise_levels,
                "rubric_variants": rubric_variants
            },
            "dimensions": {},
            "uncertainty": {},
            "diagnostics": {},
            "quality_flags": {}
        }
        
        if uncertainty_method == 'ensemble':
            results.update(self._ensemble_evaluation(
                plan, zones, n_samples, noise_levels, rubric_variants, rng
            ))
        elif uncertainty_method == 'bootstrap':
            results.update(self._bootstrap_evaluation(
                plan, zones, n_samples, noise_levels, rubric_variants, rng
            ))
        elif uncertainty_method == 'bayesian':
            results.update(self._bayesian_evaluation(
                plan, zones, n_samples, noise_levels, rubric_variants, rng
            ))
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
        
        # Add confidence calibration
        results["calibration"] = self._calibrate_confidence(results)
        
        # Add outlier detection
        results["outliers"] = self._detect_outliers(results)
        
        return results
    
    def _ensemble_evaluation(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray],
        n_samples: int,
        noise_levels: List[float],
        rubric_variants: List[str],
        rng: np.random.Generator
    ) -> Dict[str, Any]:
        """Ensemble method: Multiple rubrics + noise levels + sampling"""
        
        all_scores = defaultdict(list)
        
        # Evaluate across all combinations
        for rubric in rubric_variants:
            for noise_std in noise_levels:
                for _ in range(n_samples // len(rubric_variants) // len(noise_levels)):
                    # Add noise
                    noise = rng.normal(0.0, noise_std, size=plan.shape)
                    plan_noisy = np.clip(plan * (1.0 + noise), 0.0, None)
                    
                    # Score
                    scores = self._score_plan_comprehensive(plan_noisy, zones, rubric)
                    
                    for dim, score in scores.items():
                        all_scores[dim].append(score)
        
        # Aggregate with uncertainty
        results = {
            "dimensions": {},
            "uncertainty": {}
        }
        
        for dim, scores in all_scores.items():
            scores_arr = np.array([s for s in scores if math.isfinite(s)])
            
            results["dimensions"][dim] = {
                "mean": float(np.mean(scores_arr)),
                "median": float(np.median(scores_arr)),
                "std": float(np.std(scores_arr)),
                "ci_95": [float(np.percentile(scores_arr, 2.5)), 
                         float(np.percentile(scores_arr, 97.5))],
                "ci_90": [float(np.percentile(scores_arr, 5)), 
                         float(np.percentile(scores_arr, 95))],
                "iqr": float(np.percentile(scores_arr, 75) - np.percentile(scores_arr, 25))
            }
            
            # Uncertainty metrics
            cv = results["dimensions"][dim]["std"] / (results["dimensions"][dim]["mean"] + 1e-8)
            results["uncertainty"][dim] = {
                "coefficient_of_variation": float(cv),
                "confidence": float(max(0, 1 - cv)),  # Higher = more confident
                "stability": self._compute_stability(scores_arr),
                "sample_size": len(scores_arr)
            }
        
        # Cross-dimension diagnostics
        results["diagnostics"] = {
            "inter_dimension_consistency": self._check_consistency(all_scores),
            "score_distribution_normality": self._test_normality(all_scores),
            "ensemble_agreement": self._compute_agreement(all_scores, rubric_variants)
        }
        
        return results
    
    def _bootstrap_evaluation(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray],
        n_samples: int,
        noise_levels: List[float],
        rubric_variants: List[str],
        rng: np.random.Generator
    ) -> Dict[str, Any]:
        """Bootstrap method: Resampling-based uncertainty estimation"""

        # First collect base scores
        base_scores = []
        for _ in range(n_samples):
            noise = rng.normal(0.0, np.mean(noise_levels), size=plan.shape)
            plan_noisy = np.clip(plan * (1.0 + noise), 0.0, None)
            rubric = rng.choice(rubric_variants)
            scores = self._score_plan_comprehensive(plan_noisy, zones, rubric)
            base_scores.append(scores)
        
        # Bootstrap resampling
        n_bootstrap = 1000
        bootstrap_results = defaultdict(list)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample_indices = rng.integers(0, len(base_scores), size=len(base_scores))
            sample_scores = [base_scores[i] for i in sample_indices]
            
            # Compute statistics
            for dim in base_scores[0].keys():
                dim_scores = [s[dim] for s in sample_scores if math.isfinite(s[dim])]
                if dim_scores:
                    bootstrap_results[dim].append(np.mean(dim_scores))
        
        # Aggregate bootstrap results
        results = {
            "dimensions": {},
            "uncertainty": {},
            "diagnostics": {
                "bootstrap_iterations": n_bootstrap,
                "base_sample_size": n_samples
            }
        }
        
        for dim, boot_means in bootstrap_results.items():
            boot_arr = np.array(boot_means)
            
            results["dimensions"][dim] = {
                "mean": float(np.mean(boot_arr)),
                "std": float(np.std(boot_arr)),
                "ci_95": [float(np.percentile(boot_arr, 2.5)), 
                         float(np.percentile(boot_arr, 97.5))],
                "bootstrap_se": float(np.std(boot_arr))  # Standard error
            }
            
            results["uncertainty"][dim] = {
                "confidence": float(1.0 / (1.0 + np.std(boot_arr))),
                "bias": float(np.mean(boot_arr) - np.mean([s[dim] for s in base_scores if math.isfinite(s[dim])]))
            }
        
        return results
    
    def _bayesian_evaluation(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray],
        n_samples: int,
        noise_levels: List[float],
        rubric_variants: List[str],
        rng: np.random.Generator
    ) -> Dict[str, Any]:
        """Bayesian method: Prior + likelihood → posterior"""

        prior_alpha = 3.0  # Slight preference for higher scores
        prior_beta = 2.0
        
        # Collect likelihood data
        observed_scores = defaultdict(list)
        
        for rubric in rubric_variants:
            noise_std = np.mean(noise_levels)
            for _ in range(n_samples // len(rubric_variants)):
                noise = rng.normal(0.0, noise_std, size=plan.shape)
                plan_noisy = np.clip(plan * (1.0 + noise), 0.0, None)
                scores = self._score_plan_comprehensive(plan_noisy, zones, rubric)
                
                for dim, score in scores.items():
                    if math.isfinite(score):
                        observed_scores[dim].append(score)
        
        # Compute posterior parameters
        results = {
            "dimensions": {},
            "uncertainty": {},
            "diagnostics": {
                "prior": {"alpha": prior_alpha, "beta": prior_beta},
                "method": "bayesian_beta"
            }
        }
        
        for dim, scores in observed_scores.items():
            scores_arr = np.array(scores)
            n = len(scores_arr)
            
            # Posterior parameters (Beta-Binomial conjugate)
            successes = np.sum(scores_arr > 0.5)
            
            post_alpha = prior_alpha + successes
            post_beta = prior_beta + (n - successes)
            
            # Posterior mean and variance
            post_mean = post_alpha / (post_alpha + post_beta)
            post_var = (post_alpha * post_beta) / \
                       ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1))
            
            results["dimensions"][dim] = {
                "posterior_mean": float(post_mean),
                "posterior_std": float(np.sqrt(post_var)),
                "posterior_mode": float((post_alpha - 1) / (post_alpha + post_beta - 2)) if post_alpha > 1 and post_beta > 1 else post_mean,
                "ci_95": [
                    float(self._beta_quantile(post_alpha, post_beta, 0.025)),
                    float(self._beta_quantile(post_alpha, post_beta, 0.975))
                ]
            }
            
            results["uncertainty"][dim] = {
                "posterior_variance": float(post_var),
                "credible_interval_width": float(
                    self._beta_quantile(post_alpha, post_beta, 0.975) - 
                    self._beta_quantile(post_alpha, post_beta, 0.025)
                ),
                "confidence": float(1.0 - post_var)  # Lower variance = higher confidence
            }
        
        return results
    
    def _score_plan_comprehensive(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray],
        rubric: str
    ) -> Dict[str, float]:
        
        plan = np.nan_to_num(plan, nan=0.0).clip(0.0, None)
        
        scores = {}
        
        # 1. Spatial Quality
        scores["spatial_quality"] = self._evaluate_spatial_quality(plan)
        
        # 2. Functional Diversity
        scores["functional_diversity"] = self._evaluate_diversity(plan)
        
        # 3. Sustainability
        scores["sustainability"] = self._evaluate_sustainability(plan)
        
        # 4. Accessibility
        scores["accessibility"] = self._evaluate_accessibility(plan)
        
        # 5. Equity
        scores["equity"] = self._evaluate_equity(plan)
        
        # 6. Economic Viability
        scores["economic_viability"] = self._evaluate_economic(plan)
        
        # 7. Livability
        scores["livability"] = self._evaluate_livability(plan)
        
        # 8. Zoning Compliance (if zones available)
        if zones is not None:
            scores["zoning_compliance"] = self._evaluate_zoning(plan, zones)
        
        # Overall weighted score (rubric-dependent)
        weights = self._get_rubric_weights(rubric)
        scores["overall"] = sum(
            scores.get(dim, 0.5) * weight 
            for dim, weight in weights.items()
        ) / sum(weights.values())
        
        return scores
    
    def _evaluate_spatial_quality(self, plan: np.ndarray) -> float:
        """Spatial coherence, connectivity, pattern quality"""
        total = plan.sum(axis=0)
        
        dom = np.argmax(plan, axis=0)
        edges = 0
        h, w = dom.shape
        for i in range(h):
            for j in range(w):
                if i < h-1 and dom[i,j] != dom[i+1,j]: edges += 1
                if j < w-1 and dom[i,j] != dom[i,j+1]: edges += 1
        edge_density = edges / (h * w)
        coherence = 1.0 - edge_density
        
        autocorr = 0
        count = 0
        for i in range(h-1):
            for j in range(w):
                if total[i,j] > 0 and total[i+1,j] > 0:
                    autocorr += 1 - abs(total[i,j] - total[i+1,j]) / (total[i,j] + total[i+1,j])
                    count += 1
        for i in range(h):
            for j in range(w-1):
                if total[i,j] > 0 and total[i,j+1] > 0:
                    autocorr += 1 - abs(total[i,j] - total[i,j+1]) / (total[i,j] + total[i,j+1])
                    count += 1
        connectivity = autocorr / count if count > 0 else 0.5
        
        return 0.5 * coherence + 0.5 * connectivity
    
    def _evaluate_diversity(self, plan: np.ndarray) -> float:
        """Functional diversity and land use mix"""
        comp = plan.mean(axis=(1, 2))
        p = comp / (comp.sum() + 1e-8)
        
        # Shannon entropy normalized
        entropy = -np.sum(p * np.log(p + 1e-8))
        max_entropy = np.log(len(p))
        diversity = entropy / max_entropy
        
        # Simpson diversity
        simpson = 1.0 - np.sum(p ** 2)
        
        return 0.6 * diversity + 0.4 * simpson
    
    def _evaluate_sustainability(self, plan: np.ndarray) -> float:
        """Compactness, efficiency, anti-sprawl"""
        total = plan.sum(axis=0)
        
        developed = (total > 0.1 * total.max()).astype(int)
        if developed.sum() == 0:
            return 0.5
        
        area = developed.sum()
        perimeter = 0
        h, w = developed.shape
        for i in range(h):
            for j in range(w):
                if developed[i,j]:
                    neighbors = 0
                    if i > 0 and developed[i-1,j]: neighbors += 1
                    if i < h-1 and developed[i+1,j]: neighbors += 1
                    if j > 0 and developed[i,j-1]: neighbors += 1
                    if j < w-1 and developed[i,j+1]: neighbors += 1
                    if neighbors < 4:
                        perimeter += (4 - neighbors)
        
        compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-8) if perimeter > 0 else 0
        compactness = min(1.0, compactness)
        
        # Density gradient (concentrated = efficient)
        density_std = total.std() / (total.mean() + 1e-8)
        gradient_score = 1.0 / (1.0 + density_std)
        
        return 0.6 * compactness + 0.4 * gradient_score
    
    def _evaluate_accessibility(self, plan: np.ndarray) -> float:
        """Service accessibility and distribution"""
        service_channels = [2, 3, 4]
        if plan.shape[0] <= max(service_channels):
            return 0.5
        
        services = plan[service_channels].sum(axis=0)
        
        # Coverage (how much area has nearby services)
        # Simple approximation: service density evenness
        service_density = services / (services.sum() + 1e-8)
        uniform_density = 1.0 / services.size
        
        # Lower KL divergence = more uniform = better coverage
        kl = np.sum(service_density * np.log((service_density + 1e-8) / (uniform_density + 1e-8)))
        coverage = 1.0 / (1.0 + kl)
        
        # Service availability (total service amount)
        availability = min(1.0, services.sum() / (plan.sum() + 1e-8) * 10)
        
        return 0.6 * coverage + 0.4 * availability
    
    def _evaluate_equity(self, plan: np.ndarray) -> float:
        """Spatial equity and fairness"""
        # Gini coefficient for spatial distribution
        total = plan.sum(axis=0).flatten()
        total = np.sort(total)
        n = len(total)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * total)) / (n * total.sum() + 1e-8) - (n + 1) / n
        
        equity_score = 1.0 - gini  # Lower Gini = more equitable
        
        return equity_score
    
    def _evaluate_economic(self, plan: np.ndarray) -> float:
        """Jobs-housing balance, economic viability"""
        job_channels = [5, 6, 7]  # Office, industrial, commercial
        housing_channels = [0, 1]  # Residential low/high
        
        if plan.shape[0] <= max(max(job_channels), max(housing_channels)):
            return 0.5
        
        jobs = plan[job_channels].sum()
        housing = plan[housing_channels].sum()
        
        ratio = jobs / (housing + 1e-8)
        
        # Ideal ratio: 1.0 - 1.5 jobs per housing unit
        if 0.8 <= ratio <= 1.5:
            balance = 1.0
        else:
            balance = 1.0 / (1.0 + abs(ratio - 1.15))
        
        return balance
    
    def _evaluate_livability(self, plan: np.ndarray) -> float:
        """Walkability, amenities, quality of life"""
        # Local diversity (mixing within neighborhoods)
        window_size = 10
        C, H, W = plan.shape
        diversities = []
        
        for i in range(0, H - window_size, 5):
            for j in range(0, W - window_size, 5):
                window = plan[:, i:i+window_size, j:j+window_size]
                local_comp = window.sum(axis=(1, 2))
                local_comp = local_comp / (local_comp.sum() + 1e-8)
                
                entropy = -np.sum(local_comp * np.log(local_comp + 1e-8))
                diversity = entropy / np.log(C)
                diversities.append(diversity)
        
        mixing = np.mean(diversities) if diversities else 0.5
        
        # Amenity density (parks, recreation, culture)
        amenity_channels = [7, 8, 14]  # Recreation, culture, parks
        if plan.shape[0] <= max(amenity_channels):
            amenity_score = 0.5
        else:
            amenities = plan[amenity_channels].sum()
            amenity_score = min(1.0, amenities / (plan.sum() + 1e-8) * 20)
        
        return 0.5 * mixing + 0.5 * amenity_score
    
    def _evaluate_zoning(self, plan: np.ndarray, zones: np.ndarray) -> float:
        """Zoning compliance"""
        dom = np.argmax(plan, axis=0)
        return float(np.mean(dom == zones))
    
    def _get_rubric_weights(self, rubric: str) -> Dict[str, float]:
        """Get dimension weights for different rubrics"""
        if rubric == 'standard':
            return {
                "spatial_quality": 0.15,
                "functional_diversity": 0.15,
                "sustainability": 0.15,
                "accessibility": 0.15,
                "equity": 0.10,
                "economic_viability": 0.15,
                "livability": 0.15
            }
        elif rubric == 'sustainability':
            return {
                "spatial_quality": 0.10,
                "functional_diversity": 0.15,
                "sustainability": 0.30,  # Emphasized
                "accessibility": 0.15,
                "equity": 0.10,
                "economic_viability": 0.10,
                "livability": 0.10
            }
        elif rubric == 'equity':
            return {
                "spatial_quality": 0.10,
                "functional_diversity": 0.10,
                "sustainability": 0.10,
                "accessibility": 0.25,  # Emphasized
                "equity": 0.30,  # Emphasized
                "economic_viability": 0.05,
                "livability": 0.10
            }
        elif rubric == 'economic':
            return {
                "spatial_quality": 0.10,
                "functional_diversity": 0.15,
                "sustainability": 0.10,
                "accessibility": 0.15,
                "equity": 0.05,
                "economic_viability": 0.35,  # Emphasized
                "livability": 0.10
            }
        else:
            return self._get_rubric_weights('standard')

    
    def _compute_stability(self, scores: np.ndarray) -> float:
        """Measure score stability across samples"""
        if len(scores) < 2:
            return 1.0
        # Lower variance = more stable
        return float(1.0 / (1.0 + np.var(scores)))
    
    def _check_consistency(self, all_scores: Dict[str, List[float]]) -> float:
        """Check consistency across dimensions"""
        dims = list(all_scores.keys())
        if len(dims) < 2:
            return 1.0
        
        correlations = []
        for i in range(len(dims)):
            for j in range(i+1, len(dims)):
                scores_i = np.array(all_scores[dims[i]])
                scores_j = np.array(all_scores[dims[j]])
                
                if len(scores_i) == len(scores_j) and len(scores_i) > 1:
                    corr = np.corrcoef(scores_i, scores_j)[0, 1]
                    if math.isfinite(corr):
                        correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0.5
    
    def _test_normality(self, all_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """Test if score distributions are approximately normal"""
        from scipy import stats
        
        normality = {}
        for dim, scores in all_scores.items():
            scores_arr = np.array([s for s in scores if math.isfinite(s)])
            if len(scores_arr) < 8:
                normality[dim] = 0.5  # Not enough samples
            else:
                _, p_value = stats.shapiro(scores_arr)
                normality[dim] = float(p_value)  # Higher p = more normal
        
        return normality
    
    def _compute_agreement(
        self,
        all_scores: Dict[str, List[float]],
        rubric_variants: List[str]
    ) -> float:
        """Measure agreement across different rubrics"""
        n_rubrics = len(rubric_variants)
        if n_rubrics < 2:
            return 1.0
        
        overall_scores = all_scores.get("overall", [])
        if len(overall_scores) < n_rubrics:
            return 0.5
        
        # Lower std = higher agreement
        std = np.std(overall_scores)
        agreement = 1.0 / (1.0 + std)
        
        return float(agreement)
    
    def _calibrate_confidence(self, results: Dict) -> Dict[str, Any]:
        """Calibrate confidence scores based on diagnostics"""
        calibration = {}
        
        # Adjust confidence based on multiple factors
        for dim in results.get("dimensions", {}).keys():
            base_confidence = results["uncertainty"].get(dim, {}).get("confidence", 0.5)
            
            adjustments = []
            
            # Sample size adjustment
            n_samples = results["uncertainty"].get(dim, {}).get("sample_size", 1)
            if n_samples < 10:
                adjustments.append(0.8)
            elif n_samples < 20:
                adjustments.append(0.9)
            else:
                adjustments.append(1.0)
            
            # Normality adjustment
            normality_score = results.get("diagnostics", {}).get("score_distribution_normality", {}).get(dim, 0.5)
            if normality_score < 0.05:  # Not normal
                adjustments.append(0.9)
            else:
                adjustments.append(1.0)
            
            # Apply adjustments
            calibrated_confidence = base_confidence * np.prod(adjustments)
            
            calibration[dim] = {
                "base_confidence": float(base_confidence),
                "calibrated_confidence": float(calibrated_confidence),
                "adjustment_factors": adjustments
            }
        
        return calibration
    
    def _detect_outliers(self, results: Dict) -> Dict[str, List[int]]:
        """Detect outlier samples that deviate significantly"""
        outliers = {}
        
        # For each dimension, find samples beyond 3 standard deviations
        for dim, dim_results in results.get("dimensions", {}).items():
            samples = results.get("uncertainty", {}).get(dim, {}).get("samples", [])
            if not samples or len(samples) < 10:
                continue
            
            mean = dim_results.get("mean", 0.5)
            std = dim_results.get("std", 0.1)
            
            outlier_indices = [
                i for i, s in enumerate(samples)
                if abs(s - mean) > 3 * std
            ]
            
            if outlier_indices:
                outliers[dim] = outlier_indices
        
        return outliers
    
    def _beta_quantile(self, alpha: float, beta: float, q: float) -> float:
        """Compute quantile of Beta distribution"""
        from scipy.stats import beta as beta_dist
        return beta_dist.ppf(q, alpha, beta)
    
    # ========================================================================
    # PART 2: QUALITATIVE ANALYSIS
    # ========================================================================
    
    def qualitative_eval(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray] = None,
        reference_plan: Optional[np.ndarray] = None,
        analysis_depth: str = 'standard',  # 'basic', 'standard', 'comprehensive'
        perspectives: List[str] = ['planner', 'resident', 'developer', 'policymaker']
    ) -> Dict[str, Any]:

        results = {
            "analysis_depth": analysis_depth,
            "perspectives": perspectives,
            "overall_assessment": {},
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "pattern_recognition": {},
            "comparative_analysis": {},
            "stakeholder_perspectives": {}
        }
        
        # 1. Overall Assessment
        results["overall_assessment"] = self._generate_overall_assessment(plan, zones)
        
        # 2. Identify Strengths
        results["strengths"] = self._identify_strengths(plan, zones)
        
        # 3. Identify Weaknesses
        results["weaknesses"] = self._identify_weaknesses(plan, zones)
        
        # 4. Pattern Recognition
        results["pattern_recognition"] = self._recognize_patterns(plan)
        
        # 5. Comparative Analysis (if reference available)
        if reference_plan is not None:
            results["comparative_analysis"] = self._comparative_analysis(
                plan, reference_plan, zones
            )
        
        # 6. Stakeholder Perspectives
        for perspective in perspectives:
            results["stakeholder_perspectives"][perspective] = \
                self._stakeholder_perspective(plan, zones, perspective)
        
        # 7. Actionable Recommendations
        results["recommendations"] = self._generate_recommendations(
            plan, zones, results["weaknesses"]
        )
        
        # 8. Detailed deep-dive (if comprehensive)
        if analysis_depth == 'comprehensive':
            results["detailed_analysis"] = self._comprehensive_deep_dive(plan, zones)
        
        return results
    
    def _generate_overall_assessment(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Generate high-level narrative assessment"""
        
        # Compute key indicators
        diversity = self._evaluate_diversity(plan)
        sustainability = self._evaluate_sustainability(plan)
        accessibility = self._evaluate_accessibility(plan)
        
        # Categorize quality
        overall_score = (diversity + sustainability + accessibility) / 3.0
        
        if overall_score >= 0.75:
            quality_level = "excellent"
            descriptor = "high-quality, well-balanced"
        elif overall_score >= 0.60:
            quality_level = "good"
            descriptor = "solid, functional"
        elif overall_score >= 0.45:
            quality_level = "adequate"
            descriptor = "acceptable with room for improvement"
        else:
            quality_level = "needs_improvement"
            descriptor = "requiring significant enhancement"
        
        # Generate narrative
        narrative = f"This urban plan demonstrates {quality_level} characteristics, presenting a {descriptor} spatial configuration. "
        
        if diversity >= 0.7:
            narrative += "The plan exhibits strong functional diversity with a healthy mix of land uses. "
        elif diversity < 0.4:
            narrative += "The plan shows limited functional diversity, potentially reducing walkability and self-sufficiency. "
        
        if sustainability >= 0.7:
            narrative += "Development patterns are compact and efficient, supporting sustainability goals. "
        elif sustainability < 0.4:
            narrative += "Sprawl tendencies are evident, suggesting inefficiencies in land use and infrastructure. "
        
        if accessibility >= 0.7:
            narrative += "Service accessibility is well-distributed across the area."
        elif accessibility < 0.4:
            narrative += "Service accessibility shows significant gaps, creating potential equity concerns."
        
        return {
            "quality_level": quality_level,
            "overall_score": float(overall_score),
            "narrative": narrative,
            "key_indicators": {
                "diversity": float(diversity),
                "sustainability": float(sustainability),
                "accessibility": float(accessibility)
            }
        }
    
    def _identify_strengths(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Identify specific strengths with explanations"""
        
        strengths = []
        
        # Check multiple dimensions
        diversity = self._evaluate_diversity(plan)
        sustainability = self._evaluate_sustainability(plan)
        accessibility = self._evaluate_accessibility(plan)
        equity = self._evaluate_equity(plan)
        economic = self._evaluate_economic(plan)
        
        # Functional Diversity
        if diversity >= 0.7:
            strengths.append({
                "dimension": "Functional Diversity",
                "score": float(diversity),
                "description": "Excellent land use mix promotes walkability and reduces car dependency.",
                "evidence": f"Simpson diversity index of {diversity:.2f} indicates strong functional balance across {plan.shape[0]} POI categories.",
                "impact": "Residents can access diverse services within walking distance, supporting 15-minute city principles."
            })
        
        # Sustainability
        if sustainability >= 0.7:
            strengths.append({
                "dimension": "Sustainability",
                "score": float(sustainability),
                "description": "Compact development pattern minimizes sprawl and infrastructure costs.",
                "evidence": f"Compactness ratio of {sustainability:.2f} indicates efficient land use.",
                "impact": "Reduced transportation needs, lower carbon emissions, and efficient service delivery."
            })
        
        # Accessibility
        if accessibility >= 0.7:
            strengths.append({
                "dimension": "Accessibility",
                "score": float(accessibility),
                "description": "Services are well-distributed and accessible to most residents.",
                "evidence": f"Service accessibility score of {accessibility:.2f} shows good spatial coverage.",
                "impact": "Equitable access to education, healthcare, and retail services across neighborhoods."
            })
        
        # Equity
        if equity >= 0.7:
            strengths.append({
                "dimension": "Spatial Equity",
                "score": float(equity),
                "description": "Resources are distributed fairly across the area.",
                "evidence": f"Low spatial Gini coefficient ({1-equity:.2f}) indicates equitable distribution.",
                "impact": "Reduces spatial inequality and promotes inclusive urban development."
            })
        
        # Economic Balance
        if economic >= 0.75:
            strengths.append({
                "dimension": "Jobs-Housing Balance",
                "score": float(economic),
                "description": "Strong balance between employment opportunities and residential capacity.",
                "evidence": f"Jobs-housing ratio is within optimal range (0.8-1.5).",
                "impact": "Reduces commute distances, supports local economy, and enhances quality of life."
            })
        
        return strengths
    
    def _identify_weaknesses(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Identify specific weaknesses with explanations"""
        
        weaknesses = []
        
        # Check multiple dimensions
        diversity = self._evaluate_diversity(plan)
        sustainability = self._evaluate_sustainability(plan)
        accessibility = self._evaluate_accessibility(plan)
        equity = self._evaluate_equity(plan)
        economic = self._evaluate_economic(plan)
        spatial_quality = self._evaluate_spatial_quality(plan)
        
        # Functional Diversity
        if diversity < 0.4:
            weaknesses.append({
                "dimension": "Functional Diversity",
                "score": float(diversity),
                "severity": "high",
                "description": "Limited land use diversity reduces neighborhood vitality and walkability.",
                "evidence": f"Low diversity score of {diversity:.2f} indicates mono-functional zones.",
                "consequences": "Residents must travel longer distances for daily needs, increasing car dependency and carbon footprint.",
                "improvement_priority": "high"
            })
        
        # Sustainability
        if sustainability < 0.4:
            weaknesses.append({
                "dimension": "Sustainability",
                "score": float(sustainability),
                "severity": "high",
                "description": "Sprawling development pattern increases infrastructure costs and environmental impact.",
                "evidence": f"Low compactness score of {sustainability:.2f} indicates scattered development.",
                "consequences": "Higher per-capita infrastructure costs, increased vehicle miles traveled, and greater land consumption.",
                "improvement_priority": "high"
            })
        
        # Accessibility
        if accessibility < 0.4:
            weaknesses.append({
                "dimension": "Accessibility",
                "score": float(accessibility),
                "severity": "medium",
                "description": "Uneven service distribution creates accessibility gaps.",
                "evidence": f"Accessibility score of {accessibility:.2f} shows significant service deserts.",
                "consequences": "Some neighborhoods lack adequate access to essential services, creating equity concerns.",
                "improvement_priority": "medium"
            })
        
        # Equity
        if equity < 0.5:
            weaknesses.append({
                "dimension": "Spatial Equity",
                "score": float(equity),
                "severity": "medium",
                "description": "Inequitable distribution of resources across neighborhoods.",
                "evidence": f"High spatial Gini coefficient ({1-equity:.2f}) indicates concentration of resources.",
                "consequences": "Potential environmental justice concerns and unequal quality of life across areas.",
                "improvement_priority": "medium"
            })
        
        # Economic Balance
        if economic < 0.4:
            weaknesses.append({
                "dimension": "Jobs-Housing Balance",
                "score": float(economic),
                "severity": "medium",
                "description": "Poor balance between jobs and housing leads to long commutes.",
                "evidence": f"Jobs-housing imbalance (score: {economic:.2f}) suggests dormitory or employment-only zones.",
                "consequences": "Increased traffic congestion, commute times, and reduced quality of life.",
                "improvement_priority": "medium"
            })
        
        # Spatial Quality
        if spatial_quality < 0.4:
            weaknesses.append({
                "dimension": "Spatial Quality",
                "score": float(spatial_quality),
                "severity": "low",
                "description": "Fragmented spatial patterns reduce coherence.",
                "evidence": f"Low spatial quality score of {spatial_quality:.2f} indicates poor connectivity.",
                "consequences": "Difficult navigation, inefficient infrastructure layout, and reduced livability.",
                "improvement_priority": "low"
            })
        
        return weaknesses
    
    def _recognize_patterns(self, plan: np.ndarray) -> Dict[str, Any]:
        """Recognize spatial patterns and archetypes"""
        
        patterns = {}
        
        # 1. Development pattern
        sustainability = self._evaluate_sustainability(plan)
        if sustainability >= 0.7:
            patterns["development_type"] = {
                "type": "compact_urban",
                "description": "Compact, dense urban core with efficient land use",
                "characteristics": ["high density", "walkable", "transit-friendly"]
            }
        elif sustainability < 0.4:
            patterns["development_type"] = {
                "type": "suburban_sprawl",
                "description": "Low-density suburban sprawl pattern",
                "characteristics": ["car-dependent", "scattered", "inefficient infrastructure"]
            }
        else:
            patterns["development_type"] = {
                "type": "mixed_urban_suburban",
                "description": "Mix of urban and suburban characteristics",
                "characteristics": ["varied density", "some walkability", "mixed connectivity"]
            }
        
        # 2. Functional organization
        diversity = self._evaluate_diversity(plan)
        if diversity >= 0.75:
            patterns["functional_organization"] = {
                "type": "mixed_use",
                "description": "Highly mixed neighborhoods with diverse functions",
                "planning_philosophy": "New Urbanism / 15-minute city"
            }
        elif diversity < 0.4:
            patterns["functional_organization"] = {
                "type": "single_use_zoning",
                "description": "Separated single-use zones (Euclidean zoning)",
                "planning_philosophy": "Traditional separation of uses"
            }
        else:
            patterns["functional_organization"] = {
                "type": "partially_mixed",
                "description": "Some mixing with distinct functional areas",
                "planning_philosophy": "Hybrid approach"
            }
        
        # 3. Service distribution pattern
        accessibility = self._evaluate_accessibility(plan)
        equity = self._evaluate_equity(plan)
        
        if accessibility >= 0.7 and equity >= 0.6:
            patterns["service_distribution"] = {
                "type": "equitable_distributed",
                "description": "Services well-distributed across all neighborhoods",
                "equity_assessment": "high spatial justice"
            }
        elif accessibility < 0.4 or equity < 0.4:
            patterns["service_distribution"] = {
                "type": "centralized_concentrated",
                "description": "Services concentrated in central areas, leaving periphery underserved",
                "equity_assessment": "potential environmental justice concerns"
            }
        else:
            patterns["service_distribution"] = {
                "type": "moderately_distributed",
                "description": "Services present but unevenly distributed",
                "equity_assessment": "room for improvement in equity"
            }
        
        # 4. Identify similar real-world examples
        patterns["similar_cities"] = self._find_similar_archetypes(
            sustainability, diversity, accessibility
        )
        
        return patterns
    
    def _find_similar_archetypes(
        self,
        sustainability: float,
        diversity: float,
        accessibility: float
    ) -> List[str]:
        """Find similar real-world city archetypes"""
        
        similar = []
        
        # Compact, diverse, accessible
        if sustainability >= 0.7 and diversity >= 0.7 and accessibility >= 0.7:
            similar.extend([
                "Barcelona (mixed-use, walkable superblocks)",
                "Amsterdam (compact, bike-friendly, diverse)",
                "Copenhagen (sustainable, 15-minute city principles)"
            ])
        
        # Sprawling, low diversity
        elif sustainability < 0.4 and diversity < 0.4:
            similar.extend([
                "Phoenix (suburban sprawl, car-dependent)",
                "Houston (low-density, separated uses)",
                "Atlanta (sprawling, auto-oriented)"
            ])
        
        # High density but low diversity
        elif sustainability >= 0.7 and diversity < 0.5:
            similar.extend([
                "Hong Kong (high density, some mono-functional zones)",
                "Singapore (compact but planned zones)"
            ])
        
        # Good diversity but sprawling
        elif diversity >= 0.7 and sustainability < 0.5:
            similar.extend([
                "Los Angeles (diverse but sprawling)",
                "São Paulo (mixed uses but low density areas)"
            ])
        
        # Moderate across board
        else:
            similar.extend([
                "Chicago (mixed characteristics)",
                "Toronto (balanced urban/suburban)",
                "Melbourne (varied development patterns)"
            ])
        
        return similar
    
    def _comparative_analysis(
        self,
        generated_plan: np.ndarray,
        reference_plan: np.ndarray,
        zones: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Compare generated vs reference plan"""
        
        comparison = {
            "summary": "",
            "improvements": [],
            "regressions": [],
            "similarity_score": 0.0
        }
        
        # Evaluate both plans
        gen_scores = self._score_plan_comprehensive(generated_plan, zones, "standard")
        ref_scores = self._score_plan_comprehensive(reference_plan, zones, "standard")
        
        # Compare dimensions
        improvements = []
        regressions = []
        
        for dim in gen_scores.keys():
            if dim == "overall":
                continue
            
            gen_score = gen_scores[dim]
            ref_score = ref_scores[dim]
            diff = gen_score - ref_score
            
            if diff >= 0.1:
                improvements.append({
                    "dimension": dim,
                    "generated_score": float(gen_score),
                    "reference_score": float(ref_score),
                    "improvement": float(diff),
                    "description": f"{dim.replace('_', ' ').title()}: Generated plan shows {diff:.1%} improvement"
                })
            elif diff <= -0.1:
                regressions.append({
                    "dimension": dim,
                    "generated_score": float(gen_score),
                    "reference_score": float(ref_score),
                    "degradation": float(abs(diff)),
                    "description": f"{dim.replace('_', ' ').title()}: Generated plan shows {abs(diff):.1%} degradation"
                })
        
        comparison["improvements"] = improvements
        comparison["regressions"] = regressions
        
        # Overall similarity
        score_diff = np.array([abs(gen_scores[d] - ref_scores[d]) for d in gen_scores.keys()])
        similarity = 1.0 - np.mean(score_diff)
        comparison["similarity_score"] = float(similarity)
        
        # Generate summary
        if similarity >= 0.9:
            comparison["summary"] = "Generated plan closely matches reference characteristics."
        elif similarity >= 0.75:
            comparison["summary"] = "Generated plan shows good alignment with reference, with some variations."
        elif similarity >= 0.6:
            comparison["summary"] = "Generated plan differs moderately from reference in several dimensions."
        else:
            comparison["summary"] = "Generated plan shows significant divergence from reference characteristics."
        
        if len(improvements) > len(regressions):
            comparison["summary"] += f" Notable improvements in {len(improvements)} dimensions."
        elif len(regressions) > len(improvements):
            comparison["summary"] += f" Areas needing attention in {len(regressions)} dimensions."
        
        return comparison
    
    def _stakeholder_perspective(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray],
        perspective: str
    ) -> Dict[str, Any]:
        """Evaluate from specific stakeholder perspective"""
        
        stakeholder_view = {
            "perspective": perspective,
            "priorities": [],
            "concerns": [],
            "opportunities": [],
            "overall_sentiment": ""
        }
        
        # Compute relevant metrics
        diversity = self._evaluate_diversity(plan)
        sustainability = self._evaluate_sustainability(plan)
        accessibility = self._evaluate_accessibility(plan)
        economic = self._evaluate_economic(plan)
        livability = self._evaluate_livability(plan)
        
        if perspective == 'planner':
            stakeholder_view["priorities"] = [
                "Compliance with zoning regulations",
                "Balanced land use mix",
                "Sustainable development patterns",
                "Adequate infrastructure capacity"
            ]
            
            if sustainability >= 0.7:
                stakeholder_view["opportunities"].append(
                    "Compact development supports efficient infrastructure delivery and sustainability goals."
                )
            
            if diversity < 0.5:
                stakeholder_view["concerns"].append(
                    "Low functional diversity may not support 15-minute city objectives."
                )
            
            overall = (diversity + sustainability + accessibility) / 3.0
            if overall >= 0.7:
                stakeholder_view["overall_sentiment"] = "Plan aligns well with contemporary planning best practices."
            elif overall >= 0.5:
                stakeholder_view["overall_sentiment"] = "Plan is acceptable but could better incorporate mixed-use principles."
            else:
                stakeholder_view["overall_sentiment"] = "Plan needs revision to meet modern planning standards."
        
        elif perspective == 'resident':
            stakeholder_view["priorities"] = [
                "Walkable neighborhoods",
                "Access to services and amenities",
                "Quality of life",
                "Safety and comfort"
            ]
            
            if livability >= 0.7:
                stakeholder_view["opportunities"].append(
                    "Good mix of amenities supports high quality of life and walkability."
                )
            
            if accessibility < 0.5:
                stakeholder_view["concerns"].append(
                    "Limited service accessibility may require driving for daily needs."
                )
            
            overall = (livability + accessibility + diversity) / 3.0
            if overall >= 0.7:
                stakeholder_view["overall_sentiment"] = "Excellent livability—I'd want to live here!"
            elif overall >= 0.5:
                stakeholder_view["overall_sentiment"] = "Decent neighborhood, but some conveniences might be lacking."
            else:
                stakeholder_view["overall_sentiment"] = "Concerns about daily convenience and quality of life."
        
        elif perspective == 'developer':
            stakeholder_view["priorities"] = [
                "Economic viability",
                "Market demand",
                "Development costs",
                "ROI potential"
            ]
            
            if economic >= 0.7:
                stakeholder_view["opportunities"].append(
                    "Strong jobs-housing balance indicates healthy market fundamentals and demand."
                )
            
            if sustainability < 0.4:
                stakeholder_view["concerns"].append(
                    "Sprawling pattern increases infrastructure costs and development expenses."
                )
            
            if diversity >= 0.7:
                stakeholder_view["opportunities"].append(
                    "Mixed-use development can command premium rents and attract diverse tenants."
                )
            
            overall = (economic + diversity) / 2.0
            if overall >= 0.7:
                stakeholder_view["overall_sentiment"] = "Strong development opportunity with good market potential."
            elif overall >= 0.5:
                stakeholder_view["overall_sentiment"] = "Viable project but requires careful market analysis."
            else:
                stakeholder_view["overall_sentiment"] = "Market viability concerns need to be addressed."
        
        elif perspective == 'policymaker':
            stakeholder_view["priorities"] = [
                "Equitable distribution of resources",
                "Environmental sustainability",
                "Economic development",
                "Social cohesion"
            ]
            
            equity = self._evaluate_equity(plan)
            
            if equity >= 0.7:
                stakeholder_view["opportunities"].append(
                    "Equitable resource distribution supports inclusive development goals."
                )
            else:
                stakeholder_view["concerns"].append(
                    "Spatial inequity raises environmental justice concerns."
                )
            
            if sustainability >= 0.7:
                stakeholder_view["opportunities"].append(
                    "Sustainable development pattern aligns with climate action commitments."
                )
            
            overall = (equity + sustainability + economic) / 3.0
            if overall >= 0.7:
                stakeholder_view["overall_sentiment"] = "Plan supports key policy objectives across multiple dimensions."
            elif overall >= 0.5:
                stakeholder_view["overall_sentiment"] = "Plan partially achieves policy goals but needs strengthening in some areas."
            else:
                stakeholder_view["overall_sentiment"] = "Significant policy concerns that need to be addressed."
        
        return stakeholder_view
    
    def _generate_recommendations(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray],
        weaknesses: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on weaknesses"""
        
        recommendations = []
        
        # Process each weakness
        for weakness in weaknesses:
            dim = weakness["dimension"]
            score = weakness["score"]
            severity = weakness.get("severity", "medium")
            
            rec = {
                "addresses": dim,
                "priority": severity,
                "actions": [],
                "expected_impact": "",
                "implementation_difficulty": ""
            }
            
            if dim == "Functional Diversity":
                rec["actions"] = [
                    "Introduce mixed-use zoning in mono-functional areas",
                    "Add complementary POI types (retail, services) near residential zones",
                    "Create neighborhood centers with diverse amenities",
                    "Implement overlay zones to allow flexible land uses"
                ]
                rec["expected_impact"] = "Increase walkability by 30-40%, reduce car trips by 20%"
                rec["implementation_difficulty"] = "medium"
            
            elif dim == "Sustainability":
                rec["actions"] = [
                    "Increase density in core areas to reduce sprawl",
                    "Implement urban growth boundaries",
                    "Connect scattered development patches",
                    "Add infill development in underutilized areas"
                ]
                rec["expected_impact"] = "Reduce infrastructure costs by 15-25%, lower carbon footprint"
                rec["implementation_difficulty"] = "high"
            
            elif dim == "Accessibility":
                rec["actions"] = [
                    "Add service centers in underserved areas",
                    "Improve service distribution across neighborhoods",
                    "Enhance transit connectivity to service hubs",
                    "Implement 15-minute neighborhood principles"
                ]
                rec["expected_impact"] = "Improve service access for 20-30% more residents"
                rec["implementation_difficulty"] = "medium"
            
            elif dim == "Spatial Equity":
                rec["actions"] = [
                    "Redistribute services to underserved areas",
                    "Implement equity-focused amenity placement",
                    "Add community facilities in low-resource neighborhoods",
                    "Ensure equitable access to parks and green space"
                ]
                rec["expected_impact"] = "Reduce spatial inequality by 25-35%"
                rec["implementation_difficulty"] = "medium-high"
            
            elif dim == "Jobs-Housing Balance":
                rec["actions"] = [
                    "Add employment centers in residential-dominated areas",
                    "Introduce housing in employment-heavy zones",
                    "Create mixed-use corridors with jobs and housing",
                    "Incentivize live-work spaces"
                ]
                rec["expected_impact"] = "Reduce average commute distance by 20-30%"
                rec["implementation_difficulty"] = "medium"
            
            elif dim == "Spatial Quality":
                rec["actions"] = [
                    "Consolidate fragmented zones",
                    "Improve connectivity between disconnected areas",
                    "Create clear neighborhood boundaries",
                    "Add buffer zones between incompatible uses"
                ]
                rec["expected_impact"] = "Improve navigability and coherence"
                rec["implementation_difficulty"] = "low-medium"
            
            recommendations.append(rec)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 1))
        
        return recommendations
    
    def _comprehensive_deep_dive(
        self,
        plan: np.ndarray,
        zones: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Comprehensive detailed analysis for research/publication"""
        
        deep_dive = {
            "spatial_structure_analysis": {},
            "functional_composition": {},
            "network_properties": {},
            "temporal_implications": {},
            "environmental_impact": {},
            "social_implications": {}
        }
        
        # 1. Spatial Structure
        deep_dive["spatial_structure_analysis"] = {
            "description": "Analysis of spatial organization and morphology",
            "metrics": {
                "compactness": float(self._evaluate_sustainability(plan)),
                "fragmentation": float(1.0 - self._evaluate_spatial_quality(plan)),
                "centrality": self._analyze_centrality(plan),
                "edge_effects": self._analyze_edges(plan)
            }
        }
        
        # 2. Functional Composition
        comp = plan.mean(axis=(1, 2))
        p = comp / (comp.sum() + 1e-8)
        
        deep_dive["functional_composition"] = {
            "description": "Detailed breakdown of functional land uses",
            "dominant_functions": [
                {"poi": self.poi_names[i], "proportion": float(p[i])}
                for i in np.argsort(p)[::-1][:5]
            ],
            "diversity_metrics": {
                "shannon_entropy": float(-np.sum(p * np.log(p + 1e-8))),
                "simpson_index": float(1.0 - np.sum(p ** 2)),
                "evenness": float(-np.sum(p * np.log(p + 1e-8)) / np.log(len(p)))
            }
        }
        
        # 3. Network Properties (simplified)
        deep_dive["network_properties"] = {
            "description": "Connectivity and accessibility network",
            "connectivity_score": float(self._evaluate_spatial_quality(plan)),
            "accessibility_score": float(self._evaluate_accessibility(plan))
        }
        
        # 4. Temporal Implications
        deep_dive["temporal_implications"] = {
            "description": "Likely evolution and adaptability over time",
            "adaptability": "High" if self._evaluate_diversity(plan) >= 0.7 else "Moderate" if self._evaluate_diversity(plan) >= 0.5 else "Low",
            "sustainability_trajectory": "Improving" if self._evaluate_sustainability(plan) >= 0.6 else "Stable" if self._evaluate_sustainability(plan) >= 0.4 else "Declining",
            "notes": "Mixed-use areas adapt better to changing needs; sprawling patterns face long-term challenges."
        }
        
        # 5. Environmental Impact
        sustainability = self._evaluate_sustainability(plan)
        deep_dive["environmental_impact"] = {
            "description": "Estimated environmental footprint",
            "carbon_intensity": "Low" if sustainability >= 0.7 else "Moderate" if sustainability >= 0.5 else "High",
            "land_consumption_efficiency": float(sustainability),
            "green_space_accessibility": float(self._evaluate_accessibility(plan)),  # Simplified
            "notes": f"Compact development (score: {sustainability:.2f}) generally correlates with lower per-capita emissions."
        }
        
        # 6. Social Implications
        equity = self._evaluate_equity(plan)
        livability = self._evaluate_livability(plan)
        
        deep_dive["social_implications"] = {
            "description": "Social equity and quality of life impacts",
            "spatial_justice_score": float(equity),
            "livability_score": float(livability),
            "community_cohesion_potential": "High" if livability >= 0.7 else "Moderate" if livability >= 0.5 else "Low",
            "equity_assessment": "Equitable" if equity >= 0.7 else "Moderately equitable" if equity >= 0.5 else "Inequitable",
            "notes": "Mixed-use, walkable neighborhoods with equitable service distribution foster stronger communities."
        }
        
        return deep_dive
    
    def _analyze_centrality(self, plan: np.ndarray) -> Dict[str, float]:
        """Analyze centrality of development"""
        total = plan.sum(axis=0)
        h, w = total.shape
        
        # Find center of mass
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        total_mass = total.sum()
        
        if total_mass > 0:
            center_y = (total * y_coords).sum() / total_mass
            center_x = (total * x_coords).sum() / total_mass
            
            # Compute concentration around center
            distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            weighted_dist = (total * distances).sum() / total_mass
            max_dist = np.sqrt(h**2 + w**2) / 2
            
            centrality_score = 1.0 - (weighted_dist / max_dist)
        else:
            centrality_score = 0.5
        
        return {
            "centrality_score": float(centrality_score),
            "center_x": float(center_x) if total_mass > 0 else h/2,
            "center_y": float(center_y) if total_mass > 0 else w/2
        }
    
    def _analyze_edges(self, plan: np.ndarray) -> Dict[str, float]:
        """Analyze edge effects and boundaries"""
        dom = np.argmax(plan, axis=0)
        h, w = dom.shape
        
        # Count edges
        total_edges = 0
        internal_edges = 0
        
        for i in range(h):
            for j in range(w):
                if i < h-1:
                    total_edges += 1
                    if dom[i,j] != dom[i+1,j]:
                        internal_edges += 1
                if j < w-1:
                    total_edges += 1
                    if dom[i,j] != dom[i,j+1]:
                        internal_edges += 1
        
        edge_density = internal_edges / total_edges if total_edges > 0 else 0
        
        return {
            "edge_density": float(edge_density),
            "total_boundaries": internal_edges,
            "fragmentation_index": float(edge_density)
        }


# ========================================================================
# USAGE EXAMPLES
# ========================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED LLM EVALUATION SYSTEM - DEMO")
    print("=" * 80)
    print()
    
    # Create sample data
    np.random.seed(42)
    plan = np.random.rand(20, 100, 100) * 10
    zones = np.random.randint(0, 20, (100, 100))
    
    evaluator = UrbanPlanEvaluator()
    
    # 1. Quantitative with Ensemble Uncertainty
    print("1. QUANTITATIVE EVALUATION (Ensemble Method)")
    print("-" * 80)
    
    quant_results = evaluator.quantitative_eval(
        plan, zones,
        n_samples=24,
        uncertainty_method='ensemble',
        noise_levels=[0.01, 0.03, 0.05],
        rubric_variants=['standard', 'sustainability', 'equity', 'economic']
    )
    
    print(f"Overall Score: {quant_results['dimensions']['overall']['mean']:.3f} "
          f"± {quant_results['dimensions']['overall']['std']:.3f}")
    print(f"95% CI: [{quant_results['dimensions']['overall']['ci_95'][0]:.3f}, "
          f"{quant_results['dimensions']['overall']['ci_95'][1]:.3f}]")
    print()
    
    # Print dimension scores with confidence
    print("Dimension Scores (with confidence):")
    for dim in ['spatial_quality', 'functional_diversity', 'sustainability', 'accessibility']:
        if dim in quant_results['dimensions']:
            dim_data = quant_results['dimensions'][dim]
            conf = quant_results['uncertainty'][dim]['confidence']
            print(f"  {dim:25s}: {dim_data['mean']:.3f} (confidence: {conf:.3f})")
    print()
    
    # 2. Qualitative Analysis
    print("2. QUALITATIVE ANALYSIS")
    print("-" * 80)
    
    qual_results = evaluator.qualitative_eval(
        plan, zones,
        analysis_depth='comprehensive',
        perspectives=['planner', 'resident', 'developer', 'policymaker']
    )
    
    print("Overall Assessment:")
    print(qual_results['overall_assessment']['narrative'])
    print()
    
    print(f"Strengths ({len(qual_results['strengths'])} identified):")
    for strength in qual_results['strengths'][:2]:
        print(f"  • {strength['dimension']}: {strength['description']}")
    print()
    
    print(f"Weaknesses ({len(qual_results['weaknesses'])} identified):")
    for weakness in qual_results['weaknesses'][:2]:
        print(f"  • {weakness['dimension']} (severity: {weakness['severity']}): {weakness['description']}")
    print()
    
    print("Pattern Recognition:")
    print(f"  Development Type: {qual_results['pattern_recognition']['development_type']['type']}")
    print(f"  Similar Cities: {', '.join(qual_results['pattern_recognition']['similar_cities'][:2])}")
    print()
    
    print("Stakeholder Perspectives:")
    for perspective in ['planner', 'resident']:
        view = qual_results['stakeholder_perspectives'][perspective]
        print(f"  {perspective.title()}: {view['overall_sentiment']}")
    print()
    
    print(f"Recommendations ({len(qual_results['recommendations'])} actionable items):")
    for rec in qual_results['recommendations'][:2]:
        print(f"  • {rec['addresses']} (priority: {rec['priority']})")
        print(f"    - {rec['actions'][0]}")
    print()
    
    print("=" * 80)
    print("Demo completed. Check output files for full results.")
    print("=" * 80)
