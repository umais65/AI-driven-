import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'chat_screen.dart';
import 'supported_classes_dialog.dart';

class ResultsScreen extends StatefulWidget {
  final Map<String, dynamic> predictionData;
  final XFile originalImage;
  final String serverUrl;

  const ResultsScreen({
    super.key,
    required this.predictionData,
    required this.originalImage,
    required this.serverUrl,
  });

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  bool _showHeatmap = false; // State to toggle between original image and Grad-CAM heatmap
  int _selectedPredictionIndex = 0;
  late List<dynamic> _predictions;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _predictions = widget.predictionData['top_predictions'] ?? [widget.predictionData];
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    
    // Fetch active prediction data
    final activePred = _predictions[_selectedPredictionIndex];
    final String plant = activePred['plant_species'] ?? 'Unknown Plant';
    final String health = activePred['health_status'] ?? 'Unknown Condition';
    final double confidence = (activePred['confidence'] is int) 
        ? (activePred['confidence'] as int).toDouble() 
        : (activePred['confidence'] ?? 0.0);
    
    final String organic = activePred['organic_treatment'] ?? 'No organic advice available.';
    final String chemical = activePred['chemical_treatment'] ?? 'No chemical advice available.';
    final String prevention = activePred['prevention'] ?? 'No preventive advice available.';
    final String detailedAnalysis = activePred['detailed_analysis'] ?? 'No additional detailed database reference available.';
    final String classRaw = activePred['class_raw'] ?? 'unknown';

    // Decode Grad-CAM image (uses base64 from server, fallback to original)
    final String heatmapBase64 = widget.predictionData['heatmap_image_base64'] ?? '';
    final ImageProvider heatmapImage = heatmapBase64.isNotEmpty
        ? MemoryImage(base64Decode(heatmapBase64)) as ImageProvider
        : (kIsWeb
            ? NetworkImage(widget.originalImage.path) as ImageProvider
            : FileImage(File(widget.originalImage.path)) as ImageProvider);

    // Check if the active prediction plant is healthy
    final bool isHealthy = health.toLowerCase() == 'healthy' || 
                           health.toLowerCase().contains('mature healthy') || 
                           health.toLowerCase().contains('young healthy');

    final bool isLowConfidence = confidence < 0.50;

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Diagnostic Report',
          style: GoogleFonts.outfit(fontWeight: FontWeight.bold),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.library_books),
            tooltip: 'View Trained Dataset Catalog',
            onPressed: () {
              showDialog(
                context: context,
                builder: (context) => SupportedClassesDialog(serverUrl: widget.serverUrl),
              );
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 1. Plant Info & Health Status Cards
              Row(
                children: [
                  Expanded(
                    child: Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: colors.primary.withOpacity(0.08),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: colors.primary.withOpacity(0.15)),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('PLANT SPECIES', style: TextStyle(fontSize: 10, color: Colors.grey, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 4),
                          Text(
                            plant,
                            style: GoogleFonts.outfit(fontSize: 18, fontWeight: FontWeight.bold, color: colors.primary),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: isHealthy 
                            ? Colors.green.withOpacity(0.08) 
                            : Colors.orange.withOpacity(0.08),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(
                          color: isHealthy 
                              ? Colors.green.withOpacity(0.2) 
                              : Colors.orange.withOpacity(0.2),
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('HEALTH STATUS', style: TextStyle(fontSize: 10, color: Colors.grey, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 4),
                          Text(
                            health,
                            style: GoogleFonts.outfit(
                              fontSize: 18, 
                              fontWeight: FontWeight.bold, 
                              color: isHealthy ? Colors.green[700] : Colors.orange[850],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
              
              // Raw Dataset Class Identifier
              Center(
                child: Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(
                    'Trained Dataset Class: $classRaw',
                    style: const TextStyle(fontSize: 11, color: Colors.grey, fontStyle: FontStyle.italic),
                  ),
                ),
              ),

              // Low-Confidence Warning Banner
              if (isLowConfidence)
                Container(
                  margin: const EdgeInsets.only(top: 16),
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.orange.withOpacity(0.08),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.orange.withOpacity(0.2)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.warning_amber_rounded, color: Colors.orangeAccent, size: 24),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Uncertain AI Diagnosis (${(confidence * 100).toInt()}% Confidence)',
                              style: GoogleFonts.outfit(fontWeight: FontWeight.bold, fontSize: 13, color: Colors.orange[800]),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              'This leaf might not be supported in our 63 trained classes. Please check the dataset catalog or review the alternative matches below.',
                              style: TextStyle(fontSize: 11, color: Colors.orange[900], height: 1.4),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

              const SizedBox(height: 16),

              // 2. Interactive Image Area with Grad-CAM switch
              Container(
                height: 300,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.08),
                      blurRadius: 12,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                clipBehavior: Clip.antiAlias,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    AnimatedSwitcher(
                      duration: const Duration(milliseconds: 300),
                      child: _showHeatmap
                          ? Image(
                              key: const ValueKey('heatmap'),
                              image: heatmapImage,
                              fit: BoxFit.cover,
                              width: double.infinity,
                              height: double.infinity,
                            )
                          : (kIsWeb
                              ? Image.network(
                                  widget.originalImage.path,
                                  key: const ValueKey('original'),
                                  fit: BoxFit.cover,
                                  width: double.infinity,
                                  height: double.infinity,
                                )
                              : Image.file(
                                  File(widget.originalImage.path),
                                  key: const ValueKey('original'),
                                  fit: BoxFit.cover,
                                  width: double.infinity,
                                  height: double.infinity,
                                )),
                    ),

                    Positioned(
                      top: 12,
                      left: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                        decoration: BoxDecoration(
                          color: _showHeatmap ? colors.primary : Colors.black.withOpacity(0.6),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          _showHeatmap ? 'AI ATTENTION MAP (Grad-CAM)' : 'ORIGINAL IMAGE',
                          style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
                        ),
                      ),
                    ),

                    Positioned(
                      bottom: 12,
                      left: 12,
                      child: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.7),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Stack(
                              alignment: Alignment.center,
                              children: [
                                SizedBox(
                                  width: 36,
                                  height: 36,
                                  child: CircularProgressIndicator(
                                    value: confidence,
                                    backgroundColor: Colors.white24,
                                    color: colors.primary,
                                    strokeWidth: 3.5,
                                  ),
                                ),
                                Text(
                                  '${(confidence * 100).toInt()}%',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(width: 8),
                            const Text(
                              'MATCH CONFIDENCE',
                              style: TextStyle(color: Colors.white70, fontSize: 9, fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      ),
                    ),

                    Positioned(
                      bottom: 12,
                      right: 12,
                      child: FloatingActionButton.small(
                        backgroundColor: Colors.white,
                        foregroundColor: colors.primary,
                        onPressed: () {
                          setState(() {
                            _showHeatmap = !_showHeatmap;
                          });
                        },
                        tooltip: 'Toggle Heatmap',
                        child: Icon(_showHeatmap ? Icons.visibility : Icons.layers),
                      ),
                    ),
                  ],
                ),
              ),

              // Alternative Matched Candidates List
              if (_predictions.length > 1) ...[
                const SizedBox(height: 24),
                Text(
                  'Trained Dataset Matches (Top 3 Candidates)',
                  style: GoogleFonts.outfit(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: colors.onBackground,
                  ),
                ),
                const SizedBox(height: 8),
                Column(
                  children: List.generate(_predictions.length, (idx) {
                    final pred = _predictions[idx];
                    final String pSpecies = pred['plant_species'] ?? 'Unknown';
                    final String hStatus = pred['health_status'] ?? 'Unknown';
                    final double conf = (pred['confidence'] is int)
                        ? (pred['confidence'] as int).toDouble()
                        : (pred['confidence'] ?? 0.0);
                    final bool isSelected = _selectedPredictionIndex == idx;

                    return GestureDetector(
                      onTap: () {
                        setState(() {
                          _selectedPredictionIndex = idx;
                        });
                      },
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        margin: const EdgeInsets.only(bottom: 10),
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                        decoration: BoxDecoration(
                          color: isSelected ? colors.primary.withOpacity(0.06) : colors.surface,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: isSelected ? colors.primary : colors.outlineVariant.withOpacity(0.4),
                            width: isSelected ? 2 : 1,
                          ),
                        ),
                        child: Row(
                          children: [
                            Stack(
                              alignment: Alignment.center,
                              children: [
                                SizedBox(
                                  width: 32,
                                  height: 32,
                                  child: CircularProgressIndicator(
                                    value: conf,
                                    backgroundColor: colors.outlineVariant.withOpacity(0.3),
                                    color: isSelected ? colors.primary : colors.secondary,
                                    strokeWidth: 3,
                                  ),
                                ),
                                Text(
                                  '${(conf * 100).toInt()}%',
                                  style: TextStyle(
                                    fontSize: 8,
                                    fontWeight: FontWeight.bold,
                                    color: isSelected ? colors.primary : colors.onSurface,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(width: 16),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    '$pSpecies - $hStatus',
                                    style: GoogleFonts.outfit(
                                      fontSize: 14,
                                      fontWeight: FontWeight.bold,
                                      color: isSelected ? colors.primary : colors.onSurface,
                                    ),
                                  ),
                                  const SizedBox(height: 2),
                                  Text(
                                    'Class Raw: ${pred['class_raw'] ?? ''}',
                                    style: const TextStyle(fontSize: 10, color: Colors.grey),
                                  ),
                                ],
                              ),
                            ),
                            if (isSelected)
                              Icon(Icons.check_circle, color: colors.primary, size: 22)
                            else
                              Icon(Icons.circle_outlined, color: Colors.grey.withOpacity(0.5), size: 22),
                          ],
                        ),
                      ),
                    );
                  }),
                ),
              ],

              const SizedBox(height: 24),

              // 3. Treatment / Advice Header
              Text(
                'Care & Treatments',
                style: GoogleFonts.outfit(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: colors.onBackground,
                ),
              ),
              const SizedBox(height: 12),

              // 4. TabBar for Remedies
              Container(
                decoration: BoxDecoration(
                  color: colors.surface,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  children: [
                    TabBar(
                      controller: _tabController,
                      labelColor: colors.primary,
                      unselectedLabelColor: Colors.grey,
                      indicatorColor: colors.primary,
                      indicatorSize: TabBarIndicatorSize.tab,
                      tabs: const [
                        Tab(icon: Icon(Icons.spa), text: 'Organic'),
                        Tab(icon: Icon(Icons.science), text: 'Chemical'),
                        Tab(icon: Icon(Icons.shield_outlined), text: 'Prevention'),
                      ],
                    ),
                    SizedBox(
                      height: 160,
                      child: TabBarView(
                        controller: _tabController,
                        children: [
                          _buildRemedyCard(organic, colors.primary.withOpacity(0.05)),
                          _buildRemedyCard(chemical, isHealthy ? Colors.green.withOpacity(0.03) : Colors.orange.withOpacity(0.03)),
                          _buildRemedyCard(prevention, colors.secondary.withOpacity(0.05)),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Detailed RAG Insights Section
              Text(
                'Detailed Botanist Insights (RAG)',
                style: GoogleFonts.outfit(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: colors.onBackground,
                ),
              ),
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: colors.surface,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: colors.primary.withOpacity(0.15)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.auto_stories_outlined, color: colors.primary, size: 20),
                        const SizedBox(width: 8),
                        Text(
                          'Agricultural Knowledge Base Reference',
                          style: GoogleFonts.outfit(fontWeight: FontWeight.bold, fontSize: 13, color: colors.primary),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      detailedAnalysis,
                      style: TextStyle(
                        fontSize: 13,
                        height: 1.6,
                        color: colors.onSurface.withOpacity(0.9),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 28),

              // Chat with Botanist AI (Generative AI Showcase)
              ElevatedButton.icon(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => ChatScreen(
                        plantSpecies: plant,
                        healthStatus: health,
                        initialUrl: widget.serverUrl,
                      ),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: colors.primary,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  elevation: 2,
                ),
                icon: const Icon(Icons.chat),
                label: Text(
                  'Chat with Botanist AI (RAG)',
                  style: GoogleFonts.outfit(fontWeight: FontWeight.bold, fontSize: 16),
                ),
              ),
              const SizedBox(height: 12),

              // Back to Home
              OutlinedButton.icon(
                onPressed: () => Navigator.pop(context),
                style: OutlinedButton.styleFrom(
                  side: BorderSide(color: colors.primary, width: 1.5),
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                ),
                icon: const Icon(Icons.replay),
                label: Text(
                  'Diagnose Another Leaf',
                  style: GoogleFonts.outfit(fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRemedyCard(String content, Color bgColor) {
    return Container(
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(12),
      ),
      child: SingleChildScrollView(
        child: Text(
          content,
          style: const TextStyle(
            fontSize: 14,
            height: 1.5,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
    );
  }
}
