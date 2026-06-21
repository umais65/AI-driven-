import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:google_fonts/google_fonts.dart';

class SupportedClassesDialog extends StatefulWidget {
  final String serverUrl;

  const SupportedClassesDialog({super.key, required this.serverUrl});

  @override
  State<SupportedClassesDialog> createState() => _SupportedClassesDialogState();
}

class _SupportedClassesDialogState extends State<SupportedClassesDialog> {
  Map<String, List<dynamic>> _classes = {};
  Map<String, List<dynamic>> _filteredClasses = {};
  bool _isLoading = true;
  String _searchQuery = "";
  final TextEditingController _searchController = TextEditingController();

  // Robust offline fallback data listing all 63 trained classes
  static const Map<String, List<String>> _fallbackClasses = {
    "Apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"],
    "Blueberry": ["Healthy"],
    "Cherry": ["Powdery Mildew", "Healthy"],
    "Corn (Maize)": ["Cercospora Leaf Spot Gray Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"],
    "Grape": ["Black Rot", "Esca (Black Measles)", "Leaf Blight (Isariopsis Leaf Spot)", "Healthy"],
    "Orange": ["Haunglongbing (Citrus Greening)"],
    "Peach": ["Bacterial Spot", "Healthy"],
    "Pepper, Bell": ["Bacterial Spot", "Healthy"],
    "Potato": ["Early Blight", "Late Blight", "Healthy"],
    "Raspberry": ["Healthy"],
    "Soybean": ["Healthy"],
    "Squash": ["Powdery Mildew"],
    "Strawberry": ["Leaf Scorch", "Healthy"],
    "Tomato": [
      "Bacterial Spot",
      "Early Blight",
      "Late Blight",
      "Leaf Mold",
      "Septoria Leaf Spot",
      "Spider Mites Two-Spotted Spider Mite",
      "Target Spot",
      "Tomato Yellow Leaf Curl Virus",
      "Tomato Mosaic Virus",
      "Healthy"
    ],
    "Aloe Vera": ["Healthy", "Diseased", "Dried", "Chlorotic"],
    "Neem": ["Healthy", "Diseased", "Dried", "Chlorotic"],
    "Centella Asiatica": ["Healthy", "Insects", "Mild Disease"],
    "Hibiscus Rosa Sinensis": ["Healthy", "Chlorotic", "Disease"],
    "Kalanchoe Pinnata": ["Healthy", "Chlorotic", "Disease"],
    "Mikania Micrantha": ["Healthy", "Disease", "Distorted"],
    "Piper Betle": ["Healthy", "Chlorotic", "Disease"],
    "Generic": ["Healthy", "Powdery", "Rust"]
  };

  @override
  void initState() {
    super.initState();
    _fetchClasses();
  }

  Future<void> _fetchClasses() async {
    try {
      final response = await http
          .get(Uri.parse('${widget.serverUrl}/classes'))
          .timeout(const Duration(seconds: 4));
      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        final Map<String, List<dynamic>> parsed = {};
        data.forEach((key, value) {
          parsed[key] = List<dynamic>.from(value);
        });
        if (mounted) {
          setState(() {
            _classes = parsed;
            _filteredClasses = parsed;
            _isLoading = false;
          });
        }
        return;
      }
    } catch (_) {
      // Fallback silently if offline or request fails
    }

    if (mounted) {
      setState(() {
        _classes = Map<String, List<dynamic>>.from(_fallbackClasses);
        _filteredClasses = Map<String, List<dynamic>>.from(_fallbackClasses);
        _isLoading = false;
      });
    }
  }

  void _filterResults(String query) {
    setState(() {
      _searchQuery = query.trim().toLowerCase();
      if (_searchQuery.isEmpty) {
        _filteredClasses = Map<String, List<dynamic>>.from(_classes);
      } else {
        final Map<String, List<dynamic>> temp = {};
        _classes.forEach((plant, conditions) {
          final matchesPlant = plant.toLowerCase().contains(_searchQuery);
          final matchingConditions = conditions.where((cond) {
            return cond.toString().toLowerCase().contains(_searchQuery);
          }).toList();

          if (matchesPlant) {
            temp[plant] = conditions;
          } else if (matchingConditions.isNotEmpty) {
            temp[plant] = matchingConditions;
          }
        });
        _filteredClasses = temp;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
      elevation: 8,
      child: Container(
        padding: const EdgeInsets.all(20),
        width: double.infinity,
        constraints: const BoxConstraints(maxWidth: 480, maxHeight: 600),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Dialog Header
            Row(
              children: [
                Icon(Icons.library_books, color: colors.primary, size: 28),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Trained Dataset Catalog',
                        style: GoogleFonts.outfit(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: colors.onSurface,
                        ),
                      ),
                      const Text(
                        'Total 63 Supported Classes',
                        style: TextStyle(fontSize: 12, color: Colors.grey),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: () => Navigator.pop(context),
                )
              ],
            ),
            const SizedBox(height: 16),

            // Search Box
            TextField(
              controller: _searchController,
              onChanged: _filterResults,
              decoration: InputDecoration(
                prefixIcon: const Icon(Icons.search),
                hintText: 'Search plant or disease...',
                suffixIcon: _searchController.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _searchController.clear();
                          _filterResults("");
                        },
                      )
                    : null,
                contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16),
                  borderSide: BorderSide(color: colors.primary.withOpacity(0.15)),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16),
                  borderSide: BorderSide(color: colors.primary, width: 1.5),
                ),
                filled: true,
                fillColor: colors.surfaceVariant.withOpacity(0.3),
              ),
            ),
            const SizedBox(height: 16),

            // Content List
            Expanded(
              child: _isLoading
                  ? Center(
                      child: CircularProgressIndicator(color: colors.primary),
                    )
                  : _filteredClasses.isEmpty
                      ? const Center(
                          child: Padding(
                            padding: EdgeInsets.all(24.0),
                            child: Text(
                              'No matching plants or diseases found.',
                              style: TextStyle(color: Colors.grey, fontStyle: FontStyle.italic),
                            ),
                          ),
                        )
                      : ListView.builder(
                          itemCount: _filteredClasses.length,
                          itemBuilder: (context, index) {
                            final String plant = _filteredClasses.keys.elementAt(index);
                            final List<dynamic> conditions = _filteredClasses[plant]!;

                            return Container(
                              margin: const EdgeInsets.only(bottom: 12),
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: colors.surface,
                                borderRadius: BorderRadius.circular(16),
                                border: Border.all(
                                  color: colors.outlineVariant.withOpacity(0.4),
                                ),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  // Plant header
                                  Row(
                                    children: [
                                      Icon(Icons.eco, color: colors.primary.withOpacity(0.7), size: 18),
                                      const SizedBox(width: 8),
                                      Text(
                                        plant,
                                        style: GoogleFonts.outfit(
                                          fontSize: 15,
                                          fontWeight: FontWeight.bold,
                                          color: colors.onSurface,
                                        ),
                                      ),
                                    ],
                                  ),
                                  const SizedBox(height: 8),
                                  // Chips for conditions
                                  Wrap(
                                    spacing: 6,
                                    runSpacing: 6,
                                    children: conditions.map((cond) {
                                      final bool isHealthy = cond.toString().toLowerCase() == 'healthy' ||
                                          cond.toString().toLowerCase().contains('mature healthy') ||
                                          cond.toString().toLowerCase().contains('young healthy');
                                      return Container(
                                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                        decoration: BoxDecoration(
                                          color: isHealthy
                                              ? Colors.green.withOpacity(0.08)
                                              : Colors.orange.withOpacity(0.08),
                                          borderRadius: BorderRadius.circular(20),
                                          border: Border.all(
                                            color: isHealthy
                                                ? Colors.green.withOpacity(0.3)
                                                : Colors.orange.withOpacity(0.3),
                                          ),
                                        ),
                                        child: Text(
                                          cond.toString(),
                                          style: TextStyle(
                                            fontSize: 11,
                                            fontWeight: FontWeight.w600,
                                            color: isHealthy ? Colors.green[800] : Colors.orange[800],
                                          ),
                                        ),
                                      );
                                    }).toList(),
                                  ),
                                ],
                              ),
                            );
                          },
                        ),
            ),
          ],
        ),
      ),
    );
  }
}
