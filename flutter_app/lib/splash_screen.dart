import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'home_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _scaleAnimation;
  late Animation<double> _opacityAnimation;
  bool _isServerWarming = false;

  @override
  void initState() {
    super.initState();

    // Set up animations
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1800),
    );

    _scaleAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.elasticOut,
    );

    _opacityAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeIn,
    );

    _animationController.forward();

    // Warm up the backend server in the background and transition
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    // Start backend ping in background to wake it up or check status
    _warmUpBackend();

    // Hold splash screen for a premium 3-second duration
    await Future.delayed(const Duration(seconds: 3));

    if (mounted) {
      Navigator.of(context).pushReplacement(
        PageRouteBuilder(
          pageBuilder: (context, animation, secondaryAnimation) => const HomeScreen(),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            return FadeTransition(
              opacity: animation,
              child: child,
            );
          },
          transitionDuration: const Duration(milliseconds: 800),
        ),
      );
    }
  }

  Future<void> _warmUpBackend() async {
    setState(() {
      _isServerWarming = true;
    });

    try {
      // Default URL is Hugging Face Space. Ping it to pre-warm the container.
      await http.get(Uri.parse('https://umaisiss-aii-driven.hf.space')).timeout(
        const Duration(seconds: 5),
      );
    } catch (_) {
      // Fail silently, as HomeScreen will perform its own validation
    } finally {
      if (mounted) {
        setState(() {
          _isServerWarming = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Color(0xFF081C15), // Very dark emerald green
              Color(0xFF1B4332), // Deep forest green
              Color(0xFF081C15),
            ],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: SafeArea(
          child: Stack(
            children: [
              Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // Glowing Logo Shield + Leaf
                    ScaleTransition(
                      scale: _scaleAnimation,
                      child: FadeTransition(
                        opacity: _opacityAnimation,
                        child: Container(
                          width: 120,
                          height: 120,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: const Color(0xFF2D6A4F).withOpacity(0.2),
                            boxShadow: [
                              BoxShadow(
                                color: const Color(0xFF52B788).withOpacity(0.3),
                                blurRadius: 40,
                                spreadRadius: 5,
                              ),
                            ],
                          ),
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              // Outer Shield
                              Icon(
                                Icons.shield_outlined,
                                size: 90,
                                color: const Color(0xFF52B788).withOpacity(0.9),
                              ),
                              // Inner Leaf
                              const Icon(
                                Icons.eco_rounded,
                                size: 45,
                                color: Color(0xFF74C69D),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 24),
                    // Title Text
                    FadeTransition(
                      opacity: _opacityAnimation,
                      child: Text(
                        'AgriGuard AI',
                        style: GoogleFonts.outfit(
                          fontSize: 36,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                          letterSpacing: 1.2,
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    // Tagline Text
                    FadeTransition(
                      opacity: _opacityAnimation,
                      child: Text(
                        'AI-Driven Crop Protection & RAG Care',
                        style: GoogleFonts.outfit(
                          fontSize: 14,
                          fontWeight: FontWeight.w400,
                          color: const Color(0xFFB7E4C7),
                          letterSpacing: 0.8,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              // Bottom Progress/Status Indicator
              Align(
                alignment: Alignment.bottomCenter,
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 40),
                  child: FadeTransition(
                    opacity: _opacityAnimation,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const SizedBox(
                          width: 24,
                          height: 24,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF74C69D)),
                          ),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          _isServerWarming ? 'Connecting to AgriGuard backend...' : 'Initializing AgriGuard...',
                          style: GoogleFonts.outfit(
                            fontSize: 12,
                            color: Colors.white70,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
