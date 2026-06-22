import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'home_screen.dart';
import 'splash_screen.dart';

void main() {
  runApp(const AgriGuardApp());
}

class AgriGuardApp extends StatelessWidget {
  const AgriGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AgriGuard AI',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.system, // Support light/dark modes automatically
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1B4332), // Forest Green
          brightness: Brightness.light,
          primary: const Color(0xFF1B4332),
          secondary: const Color(0xFF40916C),
          background: const Color(0xFFF4F9F4),
        ),
        textTheme: GoogleFonts.outfitTextTheme(ThemeData.light().textTheme),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          centerTitle: true,
        ),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2D6A4F),
          brightness: Brightness.dark,
          primary: const Color(0xFF52B788),
          secondary: const Color(0xFF74C69D),
          background: const Color(0xFF0F1410),
          surface: const Color(0xFF17201A),
        ),
        textTheme: GoogleFonts.outfitTextTheme(ThemeData.dark().textTheme),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          centerTitle: true,
        ),
      ),
      home: const SplashScreen(),
    );
  }
}
