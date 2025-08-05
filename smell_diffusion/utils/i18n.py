"""Internationalization support for smell diffusion system."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

from .logging import SmellDiffusionLogger
from .config import get_config


class I18nManager:
    """Manages internationalization and localization."""
    
    SUPPORTED_LOCALES = {
        'en': 'English',
        'es': 'Español',
        'fr': 'Français', 
        'de': 'Deutsch',
        'ja': '日本語',
        'zh': '中文',
        'it': 'Italiano',
        'pt': 'Português',
        'ru': 'Русский',
        'ko': '한국어'
    }
    
    DEFAULT_LOCALE = 'en'
    
    def __init__(self, locale: Optional[str] = None):
        """Initialize i18n manager."""
        self.logger = SmellDiffusionLogger("i18n")
        self.locale = locale or self._detect_locale()
        self.translations = {}
        self._load_translations()
    
    def _detect_locale(self) -> str:
        """Detect system locale."""
        import os
        import locale
        
        try:
            # Try environment variables first
            system_locale = os.getenv('LANG', '').split('.')[0].replace('_', '-')
            
            # Try system locale
            if not system_locale:
                system_locale = locale.getdefaultlocale()[0]
                if system_locale:
                    system_locale = system_locale.replace('_', '-')
            
            # Extract language code
            if system_locale:
                lang_code = system_locale.split('-')[0].lower()
                if lang_code in self.SUPPORTED_LOCALES:
                    return lang_code
                    
        except Exception as e:
            self.logger.log_error("locale_detection", e)
        
        return self.DEFAULT_LOCALE
    
    @lru_cache(maxsize=1000)
    def _load_translations(self) -> None:
        """Load translation files."""
        translations_dir = Path(__file__).parent.parent / "translations"
        
        for locale in self.SUPPORTED_LOCALES:
            translation_file = translations_dir / f"{locale}.json"
            
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self.translations[locale] = json.load(f)
                except Exception as e:
                    self.logger.log_error("translation_loading", e, {"locale": locale})
                    self.translations[locale] = {}
            else:
                self.translations[locale] = {}
    
    def t(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a key to the current or specified locale."""
        target_locale = locale or self.locale
        
        # Get translation from the target locale
        translation = self._get_translation(key, target_locale)
        
        # Fallback to English if not found
        if translation == key and target_locale != self.DEFAULT_LOCALE:
            translation = self._get_translation(key, self.DEFAULT_LOCALE)
        
        # Format with parameters if provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                self.logger.log_error("translation_formatting", e, {
                    "key": key, "locale": target_locale, "kwargs": kwargs
                })
        
        return translation
    
    def _get_translation(self, key: str, locale: str) -> str:
        """Get translation for a specific key and locale."""
        if locale not in self.translations:
            return key
        
        # Support nested keys with dot notation
        parts = key.split('.')
        current = self.translations[locale]
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return key
        
        return str(current) if current is not None else key
    
    def get_localized_fragrance_notes(self, notes: Dict[str, Any], 
                                    locale: Optional[str] = None) -> Dict[str, Any]:
        """Get localized fragrance note descriptions."""
        target_locale = locale or self.locale
        localized_notes = {}
        
        for category, note_list in notes.items():
            localized_notes[category] = []
            
            for note in note_list:
                # Try to get localized note name
                localized_name = self.t(f"fragrance_notes.{note}", target_locale)
                if localized_name == f"fragrance_notes.{note}":
                    # Fallback to original if no translation
                    localized_name = note
                
                localized_notes[category].append(localized_name)
        
        return localized_notes
    
    def get_localized_safety_warnings(self, warnings: list, 
                                    locale: Optional[str] = None) -> list:
        """Get localized safety warnings."""
        target_locale = locale or self.locale
        localized_warnings = []
        
        for warning in warnings:
            # Try to get localized warning
            localized_warning = self.t(f"safety.warnings.{warning}", target_locale)
            if localized_warning == f"safety.warnings.{warning}":
                # Fallback to original if no translation
                localized_warning = warning
            
            localized_warnings.append(localized_warning)
        
        return localized_warnings
    
    def format_safety_report(self, report: Any, locale: Optional[str] = None) -> Dict[str, str]:
        """Format safety report with localized text."""
        target_locale = locale or self.locale
        
        formatted_report = {
            "title": self.t("safety.report.title", target_locale),
            "overall_score": self.t("safety.report.overall_score", target_locale, 
                                  score=report.overall_score),
            "ifra_compliant": self.t(f"safety.report.ifra_{'compliant' if report.ifra_compliant else 'non_compliant'}", 
                                   target_locale),
            "recommendations_title": self.t("safety.report.recommendations", target_locale),
            "recommendations": self.get_localized_safety_warnings(report.recommendations, target_locale)
        }
        
        return formatted_report
    
    def get_supported_locales(self) -> Dict[str, str]:
        """Get list of supported locales."""
        return self.SUPPORTED_LOCALES.copy()
    
    def set_locale(self, locale: str) -> bool:
        """Set the current locale."""
        if locale in self.SUPPORTED_LOCALES:
            self.locale = locale
            return True
        return False


# Global i18n manager instance
_i18n_manager = None


def get_i18n() -> I18nManager:
    """Get global i18n manager instance."""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    return _i18n_manager


def t(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """Convenience function for translation."""
    return get_i18n().t(key, locale, **kwargs)


class LocalizedMixin:
    """Mixin to add localization support to classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._i18n = get_i18n()
    
    def t(self, key: str, **kwargs) -> str:
        """Translate using the instance i18n manager."""
        return self._i18n.t(key, **kwargs)
    
    def localized_description(self, base_description: str, 
                            locale: Optional[str] = None) -> str:
        """Get localized description with fallback."""
        if not base_description:
            return ""
        
        # Try to find localized version
        localized = self._i18n.t(f"descriptions.{base_description}", locale)
        if localized != f"descriptions.{base_description}":
            return localized
        
        return base_description


class RegionalCompliance:
    """Handle regional compliance requirements."""
    
    REGIONS = {
        'EU': {
            'name': 'European Union',
            'regulations': ['REACH', 'CLP', 'Cosmetics Regulation'],
            'allergen_list': 'EU_ALLERGENS',
            'concentration_limits': True
        },
        'US': {
            'name': 'United States',
            'regulations': ['FDA', 'IFRA', 'RIFM'],
            'allergen_list': 'FDA_ALLERGENS',
            'concentration_limits': True
        },
        'JP': {
            'name': 'Japan',
            'regulations': ['JSQI', 'JFRL'],
            'allergen_list': 'JP_ALLERGENS',
            'concentration_limits': True
        },
        'CN': {
            'name': 'China',
            'regulations': ['NMPA'],
            'allergen_list': 'CN_ALLERGENS',
            'concentration_limits': True
        },
        'CA': {
            'name': 'Canada',
            'regulations': ['Health Canada'],
            'allergen_list': 'CA_ALLERGENS',
            'concentration_limits': True
        }
    }
    
    def __init__(self):
        """Initialize regional compliance manager."""
        self.logger = SmellDiffusionLogger("compliance")
        self._i18n = get_i18n()
    
    def check_regional_compliance(self, molecule, region: str) -> Dict[str, Any]:
        """Check compliance for a specific region."""
        if region not in self.REGIONS:
            return {
                "region": region,
                "status": "unknown",
                "message": self._i18n.t("compliance.unknown_region", region=region)
            }
        
        region_info = self.REGIONS[region]
        compliance_result = {
            "region": region,
            "region_name": region_info['name'],
            "regulations": region_info['regulations'],
            "status": "compliant",
            "issues": [],
            "warnings": []
        }
        
        try:
            # Check molecular weight constraints
            if hasattr(molecule, 'molecular_weight'):
                if molecule.molecular_weight > 1000:  # General limit
                    compliance_result["issues"].append(
                        self._i18n.t("compliance.molecular_weight_too_high")
                    )
                    compliance_result["status"] = "non_compliant"
            
            # Check allergen requirements
            if hasattr(molecule, 'get_safety_profile'):
                safety = molecule.get_safety_profile()
                if safety.allergens:
                    compliance_result["warnings"].append(
                        self._i18n.t("compliance.allergens_detected", 
                                   count=len(safety.allergens))
                    )
            
            # Region-specific checks
            if region == 'EU':
                compliance_result.update(self._check_eu_compliance(molecule))
            elif region == 'US':
                compliance_result.update(self._check_us_compliance(molecule))
            elif region == 'JP':
                compliance_result.update(self._check_jp_compliance(molecule))
                
        except Exception as e:
            self.logger.log_error("compliance_check", e, {"region": region})
            compliance_result["status"] = "error"
            compliance_result["issues"].append(
                self._i18n.t("compliance.evaluation_error")
            )
        
        return compliance_result
    
    def _check_eu_compliance(self, molecule) -> Dict[str, Any]:
        """Check EU-specific compliance requirements."""
        result = {"eu_specific": True}
        
        # REACH compliance
        if hasattr(molecule, 'smiles'):
            # In real implementation, would check against REACH database
            result["reach_registered"] = "unknown"
        
        return result
    
    def _check_us_compliance(self, molecule) -> Dict[str, Any]:
        """Check US-specific compliance requirements."""
        result = {"us_specific": True}
        
        # FDA GRAS status
        result["fda_status"] = "requires_review"
        
        return result
    
    def _check_jp_compliance(self, molecule) -> Dict[str, Any]:
        """Check Japan-specific compliance requirements."""
        result = {"jp_specific": True}
        
        # JSQI compliance
        result["jsqi_listed"] = "unknown"
        
        return result
    
    def get_global_compliance_summary(self, molecule) -> Dict[str, Any]:
        """Get compliance summary for all regions."""
        summary = {
            "molecule_id": getattr(molecule, 'smiles', 'unknown'),
            "regions": {},
            "overall_status": "compliant",
            "critical_issues": [],
            "recommendations": []
        }
        
        for region_code in self.REGIONS:
            region_compliance = self.check_regional_compliance(molecule, region_code)
            summary["regions"][region_code] = region_compliance
            
            if region_compliance["status"] == "non_compliant":
                summary["overall_status"] = "non_compliant"
                summary["critical_issues"].extend(region_compliance.get("issues", []))
        
        # Add general recommendations
        if summary["critical_issues"]:
            summary["recommendations"].append(
                self._i18n.t("compliance.recommend_reformulation")
            )
        
        return summary


# Global compliance manager
_compliance_manager = None


def get_compliance() -> RegionalCompliance:
    """Get global compliance manager instance."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = RegionalCompliance()
    return _compliance_manager