---
license: mit
task_categories:
  - text-generation
  - text-classification
language:
  - en
tags:
  - security
  - vulnerability
  - ios
  - swift
  - objective-c
  - code
  - cybersecurity
  - owasp
  - mobile-security
  - cwe
size_categories:
  - n<1K
pretty_name: iOS Security Vulnerabilities (Swift & Objective-C)
---

# iOS Security Vulnerabilities Dataset (Swift & Objective-C)

A comprehensive dataset of **27 real-world iOS security vulnerability patterns** in Swift and Objective-C, covering all **OWASP Mobile Top 10 (2024)** categories with vulnerable code, secure fixes, attack scenarios, and detection guidance.

## 🎯 Purpose

This is the **first dedicated iOS/Swift/Objective-C security vulnerability dataset** on Hugging Face. While existing datasets (TitanVul, DiverseVul, CleanVul) focus on C/C++/Java/Python, none cover iOS-specific security patterns. This dataset fills that gap.

**Use cases:**
- 🤖 Train LLMs to detect iOS security vulnerabilities
- 📚 Security training and education for iOS developers
- 🔍 Build static analysis rules for Swift/Objective-C
- 🛡️ Benchmark code security models on mobile-specific patterns
- 💡 Reference guide for iOS security best practices

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total examples | 27 |
| Train split | 21 |
| Test split | 6 |
| Languages | Swift (20), Objective-C (7) |
| OWASP categories | All 10 Mobile Top 10 |
| Unique CWE IDs | 20 |
| Severity levels | CRITICAL (4), HIGH (15), MEDIUM (8) |

## 📋 Schema

Each example contains:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique identifier (e.g., `ios-m9-001`) |
| `language` | string | `swift` or `objective-c` |
| `cwe_id` | string | CWE identifier (e.g., `CWE-312`) |
| `cwe_name` | string | CWE name |
| `owasp_mobile` | string | OWASP Mobile Top 10 category |
| `severity` | string | `CRITICAL`, `HIGH`, or `MEDIUM` |
| `ios_framework` | string | iOS framework involved |
| `ios_api_misused` | string | Specific API that's misused |
| `vulnerability_description` | string | Detailed description of the vulnerability |
| `vulnerable_code` | string | Complete vulnerable code example |
| `secure_code` | string | Complete secure fix with best practices |
| `attack_scenario` | string | Real-world attack description |
| `fix_description` | string | Step-by-step remediation guide |
| `detection_hints` | string | How to detect this vulnerability in code |
| `references` | string | Standards and documentation references |

## 🔐 Vulnerability Categories Covered

### OWASP Mobile Top 10 (2024) Coverage

| # | Category | Examples | Key CWEs |
|---|----------|----------|----------|
| M1 | Improper Credential Usage | 2 | CWE-798 (Hardcoded credentials) |
| M2 | Inadequate Supply Chain Security | 1 | CWE-829 (Untrusted SDK) |
| M3 | Insecure Authentication | 1 | CWE-287 (Biometric bypass) |
| M4 | Insufficient Input/Output Validation | 5 | CWE-79 (XSS), CWE-89 (SQLi), CWE-134 (Format string), CWE-362 (Race condition) |
| M5 | Insecure Communication | 4 | CWE-295 (Cert validation), CWE-319 (Cleartext), CWE-829 (WebView) |
| M6 | Inadequate Privacy Controls | 3 | CWE-532 (Log exposure), CWE-200 (Screenshot/clipboard) |
| M7 | Insufficient Binary Protections | 1 | CWE-693 (Jailbreak detection bypass) |
| M8 | Security Misconfiguration | 2 | CWE-939 (URL scheme), CWE-272 (Permissions) |
| M9 | Insecure Data Storage | 6 | CWE-312 (Cleartext storage), CWE-922 (Backup exposure), CWE-522 (Keychain misuse) |
| M10 | Insufficient Cryptography | 2 | CWE-327 (Broken crypto), CWE-330 (Weak random) |

### iOS-Specific APIs Covered

- **Keychain Services** — proper vs improper accessibility settings
- **UserDefaults/NSUserDefaults** — plaintext storage of secrets
- **URLSession/NSURLSession** — SSL pinning, certificate validation
- **WKWebView** — XSS, JavaScript bridges, navigation control
- **LocalAuthentication** — biometric auth bypass via passcode fallback
- **CommonCrypto** — MD5, ECB mode, PBKDF2
- **CryptoKit** — AES-GCM, SymmetricKey, SecureRandom
- **Core Data** — encrypted storage, file protection
- **FileManager** — backup exclusion, secure file paths
- **App Transport Security** — proper ATS configuration
- **GCD/Dispatch** — thread safety with barriers
- **UIPasteboard** — clipboard expiration and local-only mode

## 🚀 Quick Start

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("Arno-MHL/ios-security-vulnerabilities-swift-objc")

# Browse examples
for example in ds["train"]:
    print(f"[{example['severity']}] {example['cwe_id']}: {example['cwe_name']}")
    print(f"  Language: {example['language']}")
    print(f"  OWASP: {example['owasp_mobile']}")
    print(f"  API Misused: {example['ios_api_misused']}")
    print()

# Filter by severity
critical = ds["train"].filter(lambda x: x["severity"] == "CRITICAL")
print(f"Critical vulnerabilities: {len(critical)}")

# Filter by language
swift_only = ds["train"].filter(lambda x: x["language"] == "swift")
objc_only = ds["train"].filter(lambda x: x["language"] == "objective-c")

# Get all CWE-312 (cleartext storage) examples
cleartext = ds["train"].filter(lambda x: x["cwe_id"] == "CWE-312")
```

## 📖 Example Entry

```json
{
  "id": "ios-m9-001",
  "language": "swift",
  "cwe_id": "CWE-312",
  "cwe_name": "Cleartext Storage of Sensitive Information",
  "owasp_mobile": "M9: Insecure Data Storage",
  "severity": "HIGH",
  "ios_framework": "Foundation",
  "ios_api_misused": "UserDefaults",
  "vulnerability_description": "Storing user authentication tokens in UserDefaults without encryption...",
  "vulnerable_code": "// Full Swift code showing insecure UserDefaults usage...",
  "secure_code": "// Full Swift code showing proper Keychain Services usage...",
  "attack_scenario": "An attacker extracts an iTunes backup and reads the plist...",
  "fix_description": "Use iOS Keychain Services with kSecAttrAccessibleWhenUnlockedThisDeviceOnly...",
  "detection_hints": "Look for UserDefaults.standard.set() calls with sensitive key names...",
  "references": "OWASP Mobile Top 10 2024 M9, CWE-312"
}
```

## 🏗️ Methodology

This dataset was constructed following patterns from leading vulnerability dataset research:

- **Schema design**: Based on [SecureCode-web](https://huggingface.co/datasets/scthornton/securecode-web) (arXiv:2512.18542)
- **Vulnerability taxonomy**: OWASP Mobile Top 10 (2024) mapped to iOS-specific CWEs
- **Code patterns**: Real-world vulnerability patterns documented in iOS security research, CVE databases, and Apple security guidelines
- **Quality assurance**: Each example includes complete, compilable code (not snippets), realistic attack scenarios, and platform-specific remediation

## 📄 License

MIT License

## 📚 References

- [OWASP Mobile Top 10 (2024)](https://owasp.org/www-project-mobile-top-10/)
- [Apple Security Documentation](https://developer.apple.com/documentation/security)
- [CWE - Common Weakness Enumeration](https://cwe.mitre.org/)
- [TitanVul Dataset](https://arxiv.org/abs/2507.21817)
- [SecureCode v2.0](https://arxiv.org/abs/2512.18542)

## ✏️ Citation

```bibtex
@dataset{ios_security_vulnerabilities_2025,
  title={iOS Security Vulnerabilities Dataset (Swift & Objective-C)},
  author={Arno-MHL},
  year={2025},
  url={https://huggingface.co/datasets/Arno-MHL/ios-security-vulnerabilities-swift-objc}
}
```
