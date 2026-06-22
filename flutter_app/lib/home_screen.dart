import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:google_fonts/google_fonts.dart';
import 'results_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  XFile? _selectedImage;
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false;
  String _loadingMessage = "Uploading image...";
  
  String _baseUrl = (kIsWeb || (!kIsWeb && (Platform.isWindows || Platform.isMacOS || Platform.isLinux)))
      ? "http://localhost:8000"
      : "http://10.0.2.2:8000";
  bool _isServerOnline = false;
  Timer? _statusTimer;

  @override
  void initState() {
    super.initState();
    _checkServerStatus();
    // Periodically check server status every 10 seconds
    _statusTimer = Timer.periodic(const Duration(seconds: 10), (timer) {
      _checkServerStatus();
    });
  }

  @override
  void dispose() {
    _statusTimer?.cancel();
    super.dispose();
  }

  Future<void> _checkServerStatus() async {
    try {
      final response = await http.get(Uri.parse(_baseUrl)).timeout(const Duration(seconds: 3));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _isServerOnline = data['status'] == 'online';
        });
      } else {
        setState(() {
          _isServerOnline = false;
        });
      }
    } catch (_) {
      setState(() {
        _isServerOnline = false;
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(source: source);
    if (image != null) {
      setState(() {
        _selectedImage = image;
      });
    }
  }

  void _showSettingsDialog() {
    final controller = TextEditingController(text: _baseUrl);
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Backend API Settings'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Enter your FastAPI server URL. Use http://10.0.2.2:8000 for Android emulator, or http://<PC-IP>:8000 for real phone.',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'Server URL',
                hintText: 'http://192.168.x.x:8000',
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              String url = controller.text.trim();
              if (url.endsWith('/')) {
                url = url.substring(0, url.length - 1);
              }
              setState(() {
                _baseUrl = url;
              });
              Navigator.pop(context);
              _checkServerStatus();
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  Future<void> _diagnoseLeaf() async {
    if (_selectedImage == null) return;

    setState(() {
      _isLoading = true;
      _loadingMessage = "Sending leaf image to AgriShield AI...";
    });

    // Cycle messages to keep user engaged
    final messages = [
      "Analyzing leaf cell structures...",
      "Matching against 45+ plant species...",
      "Locating disease lesions...",
      "Generating Grad-CAM heatmap...",
      "Almost done..."
    ];
    int messageIndex = 0;
    
    final timer = Timer.periodic(const Duration(milliseconds: 2500), (t) {
      if (mounted && _isLoading) {
        setState(() {
          _loadingMessage = messages[messageIndex % messages.length];
          messageIndex++;
        });
      }
    });

    try {
      var request = http.MultipartRequest('POST', Uri.parse('$_baseUrl/predict'));
      final bytes = await _selectedImage!.readAsBytes();
      request.files.add(http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: _selectedImage!.name,
      ));

      var streamedResponse = await request.send().timeout(const Duration(seconds: 25));
      var response = await http.Response.fromStream(streamedResponse);

      timer.cancel();

      if (response.statusCode == 200) {
        final Map<String, dynamic> result = json.decode(response.body);
        if (mounted) {
          setState(() {
            _isLoading = false;
          });
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ResultsScreen(
                predictionData: result,
                originalImage: _selectedImage!,
                serverUrl: _baseUrl,
              ),
            ),
          );
        }
      } else {
        throw Exception("Server error: ${response.statusCode}");
      }
    } catch (e) {
      timer.cancel();
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to diagnose leaf: ${e.toString()}'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.shield, color: colors.primary, size: 28),
            const SizedBox(width: 8),
            Text(
              'AgriShield AI',
              style: GoogleFonts.outfit(fontWeight: FontWeight.bold, fontSize: 24),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _showSettingsDialog,
          ),
        ],
      ),
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Server Status Indicator
                Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: _isServerOnline
                          ? Colors.green.withOpacity(0.15)
                          : Colors.red.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: _isServerOnline ? Colors.green : Colors.red,
                        width: 1.2,
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: _isServerOnline ? Colors.green : Colors.red,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _isServerOnline ? 'Server: Connected' : 'Server: Offline',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: _isServerOnline ? Colors.green[800] : Colors.red[800],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Main Image Card
                Expanded(
                  child: GestureDetector(
                    onTap: () => _pickImage(ImageSource.gallery),
                    child: Container(
                      decoration: BoxDecoration(
                        color: colors.surface,
                        borderRadius: BorderRadius.circular(28),
                        border: Border.all(
                          color: colors.primary.withOpacity(0.15),
                          width: 2,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: colors.primary.withOpacity(0.05),
                            blurRadius: 15,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                      clipBehavior: Clip.antiAlias,
                      child: _selectedImage != null
                          ? Stack(
                              fit: StackFit.expand,
                              children: [
                                kIsWeb
                                    ? Image.network(_selectedImage!.path, fit: BoxFit.cover)
                                    : Image.file(File(_selectedImage!.path), fit: BoxFit.cover),
                                Positioned(
                                  top: 12,
                                  right: 12,
                                  child: IconButton(
                                    style: IconButton.styleFrom(
                                      backgroundColor: Colors.black.withOpacity(0.6),
                                      foregroundColor: Colors.white,
                                    ),
                                    icon: const Icon(Icons.close),
                                    onPressed: () {
                                      setState(() {
                                        _selectedImage = null;
                                      });
                                    },
                                  ),
                                ),
                              ],
                            )
                          : Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.add_photo_alternate_outlined,
                                  size: 64,
                                  color: colors.primary.withOpacity(0.6),
                                ),
                                const SizedBox(height: 16),
                                Text(
                                  'Tap here to upload leaf image',
                                  style: GoogleFonts.outfit(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w600,
                                    color: colors.onSurface,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                const Text(
                                    'or capture a leaf photo below',
                                    style: TextStyle(color: Colors.grey, fontSize: 12),
                                  ),
                              ],
                            ),
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Capture Actions
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: () => _pickImage(ImageSource.camera),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: colors.secondary.withOpacity(0.15),
                          foregroundColor: colors.secondary,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                          elevation: 0,
                        ),
                        icon: const Icon(Icons.camera_alt),
                        label: const Text('Camera'),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: () => _pickImage(ImageSource.gallery),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: colors.secondary.withOpacity(0.15),
                          foregroundColor: colors.secondary,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                          elevation: 0,
                        ),
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Gallery'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),

                // Diagnosis button
                if (_selectedImage != null)
                  ElevatedButton(
                    onPressed: _diagnoseLeaf,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: colors.primary,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 18),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      elevation: 2,
                    ),
                    child: Text(
                      'Diagnose Leaf',
                      style: GoogleFonts.outfit(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
              ],
            ),
          ),
          
          // Full-screen loading overlay
          if (_isLoading)
            Container(
              color: Colors.black.withOpacity(0.75),
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SpinKitDoubleBounce(
                      color: colors.primary,
                      size: 70.0,
                    ),
                    const SizedBox(height: 28),
                    Text(
                      _loadingMessage,
                      textAlign: TextAlign.center,
                      style: GoogleFonts.outfit(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
