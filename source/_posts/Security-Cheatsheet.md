---
title: CS161-Computer Security Midterm Cheatsheet
mathjax: true
date: 2024-10-21 17:35:08
tags:
- Computer Security
- Cheatsheet
category:
- UCB-Security
header_image:
abstract: UC-Berkeley 24FA Computer Security Midterm Cheatsheet - x86ASM, Memory Vulnerability, Cryptography & Cryptography Vulnerability
---


> Midterm Cheatsheet for [CS 161 Fall 2024 | Computer Security](https://fa24.cs161.org/)
>
> Author: [SimonXie2004.github.io](https://simonxie2004.github.io)

<img src="/images/Security-Cheatsheet/logo.png" alt="CS161 Logo" style="zoom:70%;" />

## Resources

[Download Cheatsheet (handwritten pdf)](/files/Security-Cheatsheet/Cheat_Sheet_compressed.pdf)

## Contents

1. Memory Safety
   1. Simple Topics
      1. x86 ASM & Calling Convention
      2. Buffer-Overflow Attack
      3. Integer-Overflow Attack
      4. Format String Vulnerability
      5. Off-by-One Attack
      6. VTable Overflow
   2. Advanced Topics
      1. Ret-to-libc Attack
      2. ROP Attack
      3. Ret-to-Ret Attack
      4. Ret-to-Pop Attack
      5. Ret-to-eax Attack
      6. Ret-to-`jmp` Attack
   3. Mitigations
      1. Stack Canaries
      2. ALSR (Address Layout Randomize)
      3. PAC (Pointer Authentication Code)
      4. DEP (Non Executable Pages)
2. Cryptography
   1. Confidentiality/Integrity/Authenticity
   2. Correctness/Efficiency/Security
   3. Symmetric Key Encryption / MAC
      1. AES (AES-ECB, AES-CBC, AES-CTR, AES-CFB), Stream-Cipher
      2. Hash Function, NMAC, HMAC, HMAC-DRBG
   4. Asymmetric Key Encryption / Digital Sign
      1. Diffle-Hellman Key Exchange
      2. ElGamal Encryption
      3. RSA Encryption / Signatures
   5. MISC
      1. PRNGs (Pseudo Random Generators)
      2. Trust Anchor / CA
      3. Password Storing
