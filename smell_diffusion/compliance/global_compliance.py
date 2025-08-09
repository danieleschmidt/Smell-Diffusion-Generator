"""
Global Compliance and Regulatory Framework

Comprehensive compliance system for international markets:
- Multi-regional regulatory compliance (EU, US, JP, CN, CA, AU)
- GDPR, CCPA, PDPA data protection compliance
- Cultural sensitivity and localization
- Automated compliance checking
"""

import time
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

from ..core.molecule import Molecule
from ..utils.logging import SmellDiffusionLogger
from ..utils.i18n import I18nManager


class ComplianceRegion(Enum):
    """Supported regulatory regions."""
    EU = "eu"           # European Union
    US = "us"           # United States
    JP = "jp"           # Japan
    CN = "cn"           # China
    CA = "ca"           # Canada
    AU = "au"           # Australia
    IN = "in"           # India
    BR = "br"           # Brazil
    KR = "kr"           # South Korea
    SG = "sg"           # Singapore


class ComplianceLevel(Enum):
    """Compliance assessment levels."""
    COMPLIANT = "compliant"
    RESTRICTED = "restricted"
    PROHIBITED = "prohibited"
    REQUIRES_REVIEW = "requires_review"
    UNKNOWN = "unknown"


@dataclass
class RegulatoryStandard:
    """Regulatory standard definition."""
    region: ComplianceRegion
    standard_name: str
    version: str
    effective_date: str
    requirements: Dict[str, Any]
    restricted_substances: Set[str]
    maximum_concentrations: Dict[str, float]


@dataclass
class ComplianceResult:
    """Detailed compliance assessment result."""
    region: ComplianceRegion
    overall_status: ComplianceLevel
    standard_assessments: Dict[str, ComplianceLevel]
    violations: List[str]
    warnings: List[str]
    required_actions: List[str]
    max_allowed_concentration: Optional[float]
    labeling_requirements: List[str]
    documentation_requirements: List[str]
    assessment_date: str


class GlobalComplianceChecker:
    """Comprehensive global compliance assessment system."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("global_compliance")
        self.i18n = I18nManager()
        
        # Initialize regulatory standards
        self.regulatory_standards = self._initialize_regulatory_standards()
        
        # Cultural and religious considerations
        self.cultural_restrictions = self._initialize_cultural_restrictions()
        
        # Data protection frameworks
        self.data_protection = self._initialize_data_protection()
        
    def _initialize_regulatory_standards(self) -> Dict[ComplianceRegion, RegulatoryStandard]:
        """Initialize comprehensive regulatory standards database."""
        
        standards = {}
        
        # European Union - REACH, CLP, Cosmetics Regulation
        standards[ComplianceRegion.EU] = RegulatoryStandard(
            region=ComplianceRegion.EU,
            standard_name="EU Cosmetics Regulation (EC) No 1223/2009",
            version="2023.1",
            effective_date="2023-01-01",
            requirements={
                "safety_assessment": True,
                "allergen_declaration": True,
                "cmr_substances": "restricted",
                "nanomaterials": "notification_required",
                "animal_testing": "prohibited"
            },
            restricted_substances={
                "formaldehyde", "methanol", "hydrogen_peroxide", 
                "resorcinol", "thioglycolic_acid"
            },
            maximum_concentrations={
                "benzyl_alcohol": 1.0,      # 1% max in leave-on products
                "benzyl_salicylate": 0.4,   # 0.4% max
                "citral": 1.0,              # 1% max in leave-on
                "coumarin": 0.1,            # 0.1% max in leave-on
                "eugenol": 0.1,             # 0.1% max in leave-on
                "geraniol": 1.0,            # 1% max in leave-on
                "linalool": 1.0             # 1% max in leave-on
            }
        )
        
        # United States - FDA, EPA
        standards[ComplianceRegion.US] = RegulatoryStandard(
            region=ComplianceRegion.US,
            standard_name="FDA Title 21 CFR Part 700-740",
            version="2023.2",
            effective_date="2023-03-01",
            requirements={
                "fda_registration": True,
                "gras_status": "recommended",
                "prop65_compliance": True,
                "voc_limits": True,
                "labeling_requirements": True
            },
            restricted_substances={
                "bithionol", "mercury_compounds", "vinyl_chloride",
                "zirconium_compounds", "chlorofluorocarbon"
            },
            maximum_concentrations={
                "benzyl_acetate": 5.0,      # 5% max general use
                "phenylethyl_alcohol": 10.0, # 10% max
                "methyl_anthranilate": 0.1   # 0.1% max
            }
        )
        
        # Japan - Pharmaceutical and Medical Device Act
        standards[ComplianceRegion.JP] = RegulatoryStandard(
            region=ComplianceRegion.JP,
            standard_name="Pharmaceutical and Medical Device Act",
            version="2023.1",
            effective_date="2023-04-01",
            requirements={
                "notification": True,
                "ingredient_listing": True,
                "safety_data": True,
                "manufacturing_standards": "gmp_required"
            },
            restricted_substances={
                "tar_dyes", "formaldehyde_donors", "parabens_specific"
            },
            maximum_concentrations={
                "salicylic_acid": 0.2,      # 0.2% max
                "resorcinol": 0.1,          # 0.1% max
                "phenol": 0.1               # 0.1% max
            }
        )
        
        # China - NMPA (National Medical Products Administration)
        standards[ComplianceRegion.CN] = RegulatoryStandard(
            region=ComplianceRegion.CN,
            standard_name="Cosmetic Supervision and Administration Regulation",
            version="2021.1",
            effective_date="2021-01-01",
            requirements={
                "registration": True,
                "efficacy_claims": "substantiation_required",
                "ingredient_approval": True,
                "animal_testing": "conditional"
            },
            restricted_substances={
                "kojic_acid", "arbutin", "niacinamide_high",
                "tretinoin", "hydroquinone"
            },
            maximum_concentrations={
                "niacinamide": 5.0,         # 5% max
                "salicylic_acid": 2.0,      # 2% max
                "kojic_acid": 1.0           # 1% max
            }
        )
        
        return standards
    
    def _initialize_cultural_restrictions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural and religious restrictions database."""
        
        return {
            "halal_compliance": {
                "prohibited_ingredients": {
                    "alcohol_derived", "pork_derived", "carnivorous_animal_derived"
                },
                "certification_required": True,
                "applicable_regions": ["ae", "sa", "my", "id", "pk"]
            },
            "kosher_compliance": {
                "prohibited_ingredients": {
                    "pork_derived", "shellfish_derived", "meat_dairy_mix"
                },
                "certification_required": True,
                "applicable_regions": ["il", "us_orthodox_communities"]
            },
            "vegan_requirements": {
                "prohibited_ingredients": {
                    "animal_derived", "beeswax", "carmine", "lanolin"
                },
                "certification_available": True,
                "growing_markets": ["eu", "us", "ca", "au"]
            },
            "cultural_sensitivities": {
                "india": {
                    "avoid_beef_derived": True,
                    "prefer_vegetarian": True,
                    "ayurvedic_compatibility": "preferred"
                },
                "japan": {
                    "natural_ingredients": "highly_valued",
                    "minimalist_approach": "preferred",
                    "seasonal_relevance": True
                }
            }
        }
    
    def _initialize_data_protection(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data protection compliance frameworks."""
        
        return {
            "gdpr": {
                "applicable_regions": ["eu", "uk"],
                "requirements": {
                    "lawful_basis": True,
                    "explicit_consent": True,
                    "data_minimization": True,
                    "right_to_deletion": True,
                    "data_portability": True,
                    "privacy_by_design": True
                },
                "penalties": "up_to_4_percent_revenue",
                "implementation_date": "2018-05-25"
            },
            "ccpa": {
                "applicable_regions": ["us_california"],
                "requirements": {
                    "privacy_notice": True,
                    "opt_out_right": True,
                    "deletion_right": True,
                    "non_discrimination": True
                },
                "penalties": "up_to_7500_per_violation",
                "implementation_date": "2020-01-01"
            },
            "pdpa_singapore": {
                "applicable_regions": ["sg"],
                "requirements": {
                    "consent_management": True,
                    "purpose_limitation": True,
                    "data_breach_notification": True
                },
                "penalties": "up_to_1_million_sgd",
                "implementation_date": "2014-07-02"
            },
            "lgpd": {
                "applicable_regions": ["br"],
                "requirements": {
                    "consent_based": True,
                    "data_subject_rights": True,
                    "dpo_requirement": True
                },
                "penalties": "up_to_50_million_brl",
                "implementation_date": "2020-09-18"
            }
        }
    
    def assess_comprehensive_compliance(self, molecule: Molecule, 
                                      target_regions: List[ComplianceRegion],
                                      intended_use: str = "fragrance",
                                      concentration: float = 1.0) -> Dict[ComplianceRegion, ComplianceResult]:
        """Perform comprehensive compliance assessment across multiple regions."""
        
        self.logger.logger.info(f"Starting comprehensive compliance assessment for {len(target_regions)} regions")
        
        compliance_results = {}
        
        for region in target_regions:
            try:
                result = self._assess_regional_compliance(
                    molecule, region, intended_use, concentration
                )
                compliance_results[region] = result
                
            except Exception as e:
                self.logger.log_error(f"regional_compliance_{region.value}", e)
                compliance_results[region] = ComplianceResult(
                    region=region,
                    overall_status=ComplianceLevel.UNKNOWN,
                    standard_assessments={"assessment_failed": ComplianceLevel.UNKNOWN},
                    violations=[f"Assessment failed: {str(e)}"],
                    warnings=[],
                    required_actions=["Retry compliance assessment"],
                    max_allowed_concentration=None,
                    labeling_requirements=[],
                    documentation_requirements=[],
                    assessment_date=time.strftime("%Y-%m-%d %H:%M:%S")
                )
        
        # Generate cross-regional analysis
        self._generate_cross_regional_analysis(compliance_results)
        
        return compliance_results
    
    def _assess_regional_compliance(self, molecule: Molecule, 
                                   region: ComplianceRegion,
                                   intended_use: str,
                                   concentration: float) -> ComplianceResult:
        """Assess compliance for a specific region."""
        
        if region not in self.regulatory_standards:
            return self._create_unknown_compliance_result(region)
        
        standard = self.regulatory_standards[region]
        
        # Core compliance checks
        violations = []
        warnings = []
        required_actions = []
        standard_assessments = {}
        
        # Check restricted substances
        substance_check = self._check_restricted_substances(molecule, standard)
        if substance_check["violations"]:
            violations.extend(substance_check["violations"])
            standard_assessments["restricted_substances"] = ComplianceLevel.PROHIBITED
        else:
            standard_assessments["restricted_substances"] = ComplianceLevel.COMPLIANT
        
        # Check concentration limits
        concentration_check = self._check_concentration_limits(molecule, standard, concentration)
        if concentration_check["violations"]:
            violations.extend(concentration_check["violations"])
            standard_assessments["concentration_limits"] = ComplianceLevel.RESTRICTED
        elif concentration_check["warnings"]:
            warnings.extend(concentration_check["warnings"])
            standard_assessments["concentration_limits"] = ComplianceLevel.RESTRICTED
        else:
            standard_assessments["concentration_limits"] = ComplianceLevel.COMPLIANT
        
        # Safety assessment requirements
        safety_check = self._assess_safety_requirements(molecule, standard)
        standard_assessments["safety_requirements"] = safety_check["status"]
        if safety_check["actions"]:
            required_actions.extend(safety_check["actions"])
        
        # Labeling requirements
        labeling_reqs = self._generate_labeling_requirements(molecule, standard, region)
        
        # Documentation requirements
        doc_reqs = self._generate_documentation_requirements(standard, region)
        
        # Determine overall status
        overall_status = self._determine_overall_status(standard_assessments, violations)
        
        # Calculate maximum allowed concentration
        max_concentration = self._calculate_max_concentration(molecule, standard)
        
        return ComplianceResult(
            region=region,
            overall_status=overall_status,
            standard_assessments=standard_assessments,
            violations=violations,
            warnings=warnings,
            required_actions=required_actions,
            max_allowed_concentration=max_concentration,
            labeling_requirements=labeling_reqs,
            documentation_requirements=doc_reqs,
            assessment_date=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _check_restricted_substances(self, molecule: Molecule, 
                                   standard: RegulatoryStandard) -> Dict[str, List[str]]:
        """Check for restricted substances in the molecule."""
        
        violations = []
        warnings = []
        
        # Simplified substance checking based on SMILES patterns
        smiles = molecule.smiles.lower()
        
        for restricted_substance in standard.restricted_substances:
            if self._contains_restricted_pattern(smiles, restricted_substance):
                violations.append(
                    f"Contains restricted substance: {restricted_substance} "
                    f"(prohibited in {standard.region.value.upper()})"
                )
        
        return {"violations": violations, "warnings": warnings}
    
    def _contains_restricted_pattern(self, smiles: str, substance: str) -> bool:
        """Check if SMILES contains patterns associated with restricted substances."""
        
        # Simplified pattern matching for demonstration
        restriction_patterns = {
            "formaldehyde": ["c=o", "ch2o"],
            "methanol": ["co", "ch3oh"],
            "mercury_compounds": ["hg", "[hg]"],
            "benzyl_alcohol": ["c1=cc=c(c=c1)co"],
            "phenol": ["c1=cc=cc=c1o"],
            "tar_dyes": ["n=n", "azo"]  # Simplified azo dye detection
        }
        
        if substance in restriction_patterns:
            patterns = restriction_patterns[substance]
            return any(pattern in smiles for pattern in patterns)
        
        return False
    
    def _check_concentration_limits(self, molecule: Molecule, 
                                   standard: RegulatoryStandard,
                                   concentration: float) -> Dict[str, List[str]]:
        """Check concentration limits against regulatory standards."""
        
        violations = []
        warnings = []
        
        # Check against known concentration limits
        for substance, max_conc in standard.maximum_concentrations.items():
            if self._contains_restricted_pattern(molecule.smiles.lower(), substance):
                if concentration > max_conc:
                    violations.append(
                        f"Concentration {concentration:.2f}% exceeds maximum allowed "
                        f"{max_conc:.2f}% for {substance} in {standard.region.value.upper()}"
                    )
                elif concentration > max_conc * 0.8:  # Warning at 80% of limit
                    warnings.append(
                        f"Concentration {concentration:.2f}% approaching maximum "
                        f"{max_conc:.2f}% for {substance} in {standard.region.value.upper()}"
                    )
        
        return {"violations": violations, "warnings": warnings}
    
    def _assess_safety_requirements(self, molecule: Molecule, 
                                   standard: RegulatoryStandard) -> Dict[str, Any]:
        """Assess safety-related compliance requirements."""
        
        required_actions = []
        
        # Check safety assessment requirements
        if standard.requirements.get("safety_assessment"):
            safety_profile = molecule.get_safety_profile()
            if safety_profile.score < 70:
                required_actions.append("Conduct comprehensive safety assessment")
        
        # Check allergen declaration requirements
        if standard.requirements.get("allergen_declaration"):
            if safety_profile.allergens:
                required_actions.append("Include allergen declaration on product labeling")
        
        # Determine status based on requirements
        if required_actions:
            status = ComplianceLevel.REQUIRES_REVIEW
        else:
            status = ComplianceLevel.COMPLIANT
        
        return {"status": status, "actions": required_actions}
    
    def _generate_labeling_requirements(self, molecule: Molecule, 
                                       standard: RegulatoryStandard,
                                       region: ComplianceRegion) -> List[str]:
        """Generate region-specific labeling requirements."""
        
        requirements = []
        
        # Common requirements
        requirements.append("Product name and brand")
        requirements.append("Net content/weight")
        requirements.append("Manufacturer/distributor information")
        
        # Region-specific requirements
        if region == ComplianceRegion.EU:
            requirements.extend([
                "INCI ingredient list in descending order",
                "Allergen declaration (if >0.001% in leave-on, >0.01% in rinse-off)",
                "Best before date or Period After Opening (PAO)",
                "Warnings and precautions for use",
                "Country of origin"
            ])
        elif region == ComplianceRegion.US:
            requirements.extend([
                "FDA-compliant ingredient declaration",
                "Net weight in both metric and US customary units",
                "California Prop 65 warning (if applicable)",
                "Distributed by information"
            ])
        elif region == ComplianceRegion.JP:
            requirements.extend([
                "Japanese ingredient names",
                "Manufacturing date or expiration date",
                "Usage instructions in Japanese",
                "Import notification number (for imported products)"
            ])
        elif region == ComplianceRegion.CN:
            requirements.extend([
                "Chinese ingredient names",
                "Production license number",
                "Product standard number",
                "Net content in Chinese units",
                "Efficacy claims substantiation (if applicable)"
            ])
        
        # Check for allergens requiring specific labeling
        safety_profile = molecule.get_safety_profile()
        if safety_profile.allergens:
            requirements.append(f"Allergen warning: Contains {', '.join(safety_profile.allergens)}")
        
        return requirements
    
    def _generate_documentation_requirements(self, standard: RegulatoryStandard,
                                           region: ComplianceRegion) -> List[str]:
        """Generate required documentation for compliance."""
        
        requirements = []
        
        # Common documentation
        requirements.extend([
            "Safety assessment report",
            "Product information file (PIF)",
            "Manufacturing process documentation",
            "Quality control specifications"
        ])
        
        # Region-specific documentation
        if region == ComplianceRegion.EU:
            requirements.extend([
                "Cosmetic Product Safety Report (CPSR)",
                "CPNP notification confirmation",
                "Responsible person designation",
                "Good Manufacturing Practice (GMP) compliance certificate"
            ])
        elif region == ComplianceRegion.US:
            requirements.extend([
                "FDA facility registration",
                "Adverse event reporting procedures",
                "Substantiation for any claims made",
                "Import/export documentation (if applicable)"
            ])
        elif region == ComplianceRegion.JP:
            requirements.extend([
                "Notification to PMDA",
                "Manufacturing method statement",
                "Efficacy and safety test data",
                "GMP compliance certificate"
            ])
        elif region == ComplianceRegion.CN:
            requirements.extend([
                "NMPA registration/notification",
                "Efficacy evaluation report",
                "Safety risk assessment",
                "Production license"
            ])
        
        return requirements
    
    def _determine_overall_status(self, standard_assessments: Dict[str, ComplianceLevel],
                                 violations: List[str]) -> ComplianceLevel:
        """Determine overall compliance status."""
        
        if violations:
            return ComplianceLevel.PROHIBITED
        
        if any(status == ComplianceLevel.PROHIBITED for status in standard_assessments.values()):
            return ComplianceLevel.PROHIBITED
        
        if any(status == ComplianceLevel.RESTRICTED for status in standard_assessments.values()):
            return ComplianceLevel.RESTRICTED
        
        if any(status == ComplianceLevel.REQUIRES_REVIEW for status in standard_assessments.values()):
            return ComplianceLevel.REQUIRES_REVIEW
        
        if all(status == ComplianceLevel.COMPLIANT for status in standard_assessments.values()):
            return ComplianceLevel.COMPLIANT
        
        return ComplianceLevel.UNKNOWN
    
    def _calculate_max_concentration(self, molecule: Molecule, 
                                   standard: RegulatoryStandard) -> Optional[float]:
        """Calculate maximum allowed concentration based on regulatory limits."""
        
        max_concentrations = []
        
        for substance, max_conc in standard.maximum_concentrations.items():
            if self._contains_restricted_pattern(molecule.smiles.lower(), substance):
                max_concentrations.append(max_conc)
        
        return min(max_concentrations) if max_concentrations else None
    
    def _create_unknown_compliance_result(self, region: ComplianceRegion) -> ComplianceResult:
        """Create compliance result for unknown/unsupported regions."""
        
        return ComplianceResult(
            region=region,
            overall_status=ComplianceLevel.UNKNOWN,
            standard_assessments={"unsupported_region": ComplianceLevel.UNKNOWN},
            violations=[],
            warnings=[f"Compliance assessment not available for region: {region.value}"],
            required_actions=["Consult local regulatory authorities"],
            max_allowed_concentration=None,
            labeling_requirements=["Consult local labeling requirements"],
            documentation_requirements=["Consult local documentation requirements"],
            assessment_date=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _generate_cross_regional_analysis(self, compliance_results: Dict[ComplianceRegion, ComplianceResult]):
        """Generate analysis across multiple regions."""
        
        # Identify common compliance issues
        common_violations = set()
        common_warnings = set()
        
        for result in compliance_results.values():
            if result.violations:
                common_violations.update(result.violations)
            if result.warnings:
                common_warnings.update(result.warnings)
        
        # Log cross-regional insights
        compliant_regions = [region.value for region, result in compliance_results.items() 
                           if result.overall_status == ComplianceLevel.COMPLIANT]
        
        restricted_regions = [region.value for region, result in compliance_results.items()
                            if result.overall_status == ComplianceLevel.RESTRICTED]
        
        prohibited_regions = [region.value for region, result in compliance_results.items()
                            if result.overall_status == ComplianceLevel.PROHIBITED]
        
        self.logger.logger.info(
            f"Cross-regional compliance summary: "
            f"Compliant: {len(compliant_regions)}, "
            f"Restricted: {len(restricted_regions)}, "
            f"Prohibited: {len(prohibited_regions)}"
        )
    
    def get_cultural_compliance_guidelines(self, target_markets: List[str]) -> Dict[str, Any]:
        """Get cultural and religious compliance guidelines for target markets."""
        
        guidelines = {}
        
        for market in target_markets:
            market_guidelines = {
                "halal_requirements": [],
                "kosher_requirements": [],
                "cultural_considerations": [],
                "recommended_adaptations": []
            }
            
            # Check halal requirements
            if market.lower() in ["ae", "sa", "my", "id", "pk"]:
                halal_reqs = self.cultural_restrictions["halal_compliance"]
                market_guidelines["halal_requirements"] = [
                    "Avoid alcohol-derived ingredients",
                    "Ensure no pork-derived components", 
                    "Obtain halal certification",
                    "Use halal-certified manufacturing facilities"
                ]
            
            # Check cultural considerations
            if market.lower() == "in":
                market_guidelines["cultural_considerations"] = [
                    "Prefer vegetarian/vegan formulations",
                    "Avoid beef-derived ingredients",
                    "Consider Ayurvedic ingredient compatibility",
                    "Respect religious festivals in marketing"
                ]
            elif market.lower() == "jp":
                market_guidelines["cultural_considerations"] = [
                    "Emphasize natural and minimalist approach",
                    "Consider seasonal relevance",
                    "Respect aesthetic preferences for subtlety",
                    "Align with local beauty ideals"
                ]
            
            guidelines[market] = market_guidelines
        
        return guidelines
    
    def generate_compliance_report(self, compliance_results: Dict[ComplianceRegion, ComplianceResult],
                                  output_format: str = "json") -> str:
        """Generate comprehensive compliance report."""
        
        report_data = {
            "assessment_summary": {
                "total_regions_assessed": len(compliance_results),
                "assessment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "compliant_regions": [],
                "restricted_regions": [],
                "prohibited_regions": [],
                "review_required_regions": []
            },
            "regional_details": {},
            "recommendations": [],
            "next_steps": []
        }
        
        # Analyze results by region
        for region, result in compliance_results.items():
            region_key = region.value
            report_data["regional_details"][region_key] = {
                "overall_status": result.overall_status.value,
                "violations_count": len(result.violations),
                "warnings_count": len(result.warnings),
                "required_actions_count": len(result.required_actions),
                "max_concentration": result.max_allowed_concentration,
                "assessment_details": result.standard_assessments
            }
            
            # Categorize by status
            if result.overall_status == ComplianceLevel.COMPLIANT:
                report_data["assessment_summary"]["compliant_regions"].append(region_key)
            elif result.overall_status == ComplianceLevel.RESTRICTED:
                report_data["assessment_summary"]["restricted_regions"].append(region_key)
            elif result.overall_status == ComplianceLevel.PROHIBITED:
                report_data["assessment_summary"]["prohibited_regions"].append(region_key)
            elif result.overall_status == ComplianceLevel.REQUIRES_REVIEW:
                report_data["assessment_summary"]["review_required_regions"].append(region_key)
        
        # Generate recommendations
        report_data["recommendations"] = self._generate_compliance_recommendations(compliance_results)
        
        # Format output
        if output_format.lower() == "json":
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        else:
            return self._format_text_report(report_data)
    
    def _generate_compliance_recommendations(self, compliance_results: Dict[ComplianceRegion, ComplianceResult]) -> List[str]:
        """Generate actionable compliance recommendations."""
        
        recommendations = []
        
        # Global recommendations
        prohibited_count = sum(1 for result in compliance_results.values() 
                             if result.overall_status == ComplianceLevel.PROHIBITED)
        
        if prohibited_count > 0:
            recommendations.append(
                f"CRITICAL: Product is prohibited in {prohibited_count} regions. "
                "Consider reformulation before market entry."
            )
        
        restricted_count = sum(1 for result in compliance_results.values()
                             if result.overall_status == ComplianceLevel.RESTRICTED)
        
        if restricted_count > 0:
            recommendations.append(
                f"Product has restrictions in {restricted_count} regions. "
                "Review concentration limits and usage instructions."
            )
        
        # Specific recommendations based on common issues
        all_violations = set()
        for result in compliance_results.values():
            all_violations.update(result.violations)
        
        if any("concentration" in violation.lower() for violation in all_violations):
            recommendations.append(
                "Multiple regions have concentration limit concerns. "
                "Consider reducing active ingredient concentrations."
            )
        
        if any("allergen" in violation.lower() for violation in all_violations):
            recommendations.append(
                "Allergen issues detected across regions. "
                "Ensure proper allergen declaration and consider alternative ingredients."
            )
        
        # Documentation recommendations
        review_count = sum(1 for result in compliance_results.values()
                         if result.overall_status == ComplianceLevel.REQUIRES_REVIEW)
        
        if review_count > 0:
            recommendations.append(
                f"Additional documentation required for {review_count} regions. "
                "Prepare comprehensive safety dossier."
            )
        
        return recommendations
    
    def _format_text_report(self, report_data: Dict[str, Any]) -> str:
        """Format compliance report as human-readable text."""
        
        report_lines = [
            "GLOBAL COMPLIANCE ASSESSMENT REPORT",
            "=" * 50,
            f"Assessment Date: {report_data['assessment_summary']['assessment_date']}",
            f"Regions Assessed: {report_data['assessment_summary']['total_regions_assessed']}",
            "",
            "COMPLIANCE SUMMARY:",
            f"✅ Compliant: {len(report_data['assessment_summary']['compliant_regions'])} regions",
            f"⚠️ Restricted: {len(report_data['assessment_summary']['restricted_regions'])} regions", 
            f"❌ Prohibited: {len(report_data['assessment_summary']['prohibited_regions'])} regions",
            f"📋 Review Required: {len(report_data['assessment_summary']['review_required_regions'])} regions",
            "",
            "RECOMMENDATIONS:",
        ]
        
        for i, rec in enumerate(report_data.get("recommendations", []), 1):
            report_lines.append(f"{i}. {rec}")
        
        report_lines.extend([
            "",
            "REGIONAL DETAILS:",
            "-" * 30
        ])
        
        for region, details in report_data["regional_details"].items():
            report_lines.extend([
                f"Region: {region.upper()}",
                f"Status: {details['overall_status']}",
                f"Violations: {details['violations_count']}",
                f"Warnings: {details['warnings_count']}",
                ""
            ])
        
        return "\n".join(report_lines)