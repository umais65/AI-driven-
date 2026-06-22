import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_spinkit/flutter_spinkit.dart';

class ChatScreen extends StatefulWidget {
  final String plantSpecies;
  final String healthStatus;
  final String initialUrl; // Base URL from HomeScreen

  const ChatScreen({
    super.key,
    required this.plantSpecies,
    required this.healthStatus,
    required this.initialUrl,
  });

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<Map<String, dynamic>> _messages = [];
  bool _isTyping = false;

  @override
  void initState() {
    super.initState();
    // Add welcome message from Botanist AI
    _messages.add({
      'text': "Hello! I am your AgriGuard Botanist AI. I have reviewed the diagnosis for your **${widget.plantSpecies}** (${widget.healthStatus}). How can I help you manage this disease today?",
      'isUser': false,
    });
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _sendMessage(String text) async {
    if (text.trim().isEmpty) return;

    // Capture conversation history before appending current user message
    List<Map<String, String>> historyPayload = [];
    for (var msg in _messages) {
      historyPayload.add({
        'role': msg['isUser'] ? 'user' : 'model',
        'content': msg['text'],
      });
    }

    setState(() {
      _messages.add({'text': text, 'isUser': true});
      _isTyping = true;
    });
    _messageController.clear();
    _scrollToBottom();

    try {
      final response = await http.post(
        Uri.parse('${widget.initialUrl}/chat'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'message': text,
          'history': historyPayload,
          'context_plant': widget.plantSpecies,
          'context_disease': widget.healthStatus,
        }),
      ).timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        final List<dynamic> sources = data['sources'] ?? [];
        setState(() {
          _messages.add({
            'text': data['response'] ?? "I'm sorry, I couldn't generate a response.",
            'isUser': false,
            'sources': List<String>.from(sources.map((s) => s.toString())),
          });
          _isTyping = false;
        });
      } else {
        throw Exception("Server responded with code ${response.statusCode}");
      }
    } catch (e) {
      setState(() {
        _messages.add({
          'text': "Error: Failed to connect to server. Please check your network or backend connection.",
          'isUser': false,
        });
        _isTyping = false;
      });
    }
    _scrollToBottom();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    // Quick suggestion chips based on the plant/disease
    final List<String> suggestions = [
      "How to apply organic control?",
      "Is this pesticide safe for pets?",
      "How often should I water?",
      "Can it spread to other crops?",
    ];

    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            CircleAvatar(
              backgroundColor: colors.primary.withOpacity(0.15),
              child: Icon(Icons.spa, color: colors.primary),
            ),
            const SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Botanist AI',
                  style: GoogleFonts.outfit(fontWeight: FontWeight.bold, fontSize: 18),
                ),
                Row(
                  children: [
                    Container(
                      width: 6,
                      height: 6,
                      decoration: const BoxDecoration(
                        color: Colors.green,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 4),
                    const Text(
                      'Online',
                      style: TextStyle(fontSize: 10, color: Colors.grey),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          // Diagnostic context bar
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            color: colors.primary.withOpacity(0.08),
            child: Row(
              children: [
                const Icon(Icons.info_outline, size: 16, color: Colors.grey),
                const SizedBox(width: 8),
                Expanded(
                  child: RichText(
                    text: TextSpan(
                      style: TextStyle(color: colors.onBackground, fontSize: 12),
                      children: [
                        const TextSpan(text: "Active Context: "),
                        TextSpan(
                          text: "${widget.plantSpecies} (${widget.healthStatus})",
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Messages List
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final msg = _messages[index];
                final bool isUser = msg['isUser'];
                return Align(
                  alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin: const EdgeInsets.symmetric(vertical: 6),
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                    constraints: BoxConstraints(
                      maxWidth: MediaQuery.of(context).size.width * 0.75,
                    ),
                    decoration: BoxDecoration(
                      color: isUser
                          ? colors.primary
                          : (isDark ? const Color(0xFF1D2621) : const Color(0xFFE8F1EC)),
                      borderRadius: BorderRadius.only(
                        topLeft: const Radius.circular(16),
                        topRight: const Radius.circular(16),
                        bottomLeft: Radius.circular(isUser ? 16 : 0),
                        bottomRight: Radius.circular(isUser ? 0 : 16),
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          msg['text'],
                          style: TextStyle(
                            color: isUser ? Colors.white : colors.onSurface,
                            fontSize: 14,
                            height: 1.4,
                          ),
                        ),
                        if (!isUser && msg['sources'] != null && (msg['sources'] as List).isNotEmpty) ...[
                          const SizedBox(height: 8),
                          const Divider(height: 1, thickness: 0.5, color: Colors.grey),
                          const SizedBox(height: 4),
                          Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.menu_book_outlined, size: 12, color: Colors.grey),
                              const SizedBox(width: 4),
                              Text(
                                'RAG Sources: ${(msg['sources'] as List).join(', ')}',
                                style: const TextStyle(fontSize: 10, color: Colors.grey, fontStyle: FontStyle.italic),
                              ),
                            ],
                          ),
                        ],
                      ],
                    ),
                  ),
                );
              },
            ),
          ),

          // Typing Indicator
          if (_isTyping)
            Padding(
              padding: const EdgeInsets.only(left: 20.0, bottom: 8.0),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Row(
                  children: [
                    SpinKitThreeBounce(
                      color: colors.primary,
                      size: 16.0,
                    ),
                    const SizedBox(width: 8),
                    const Text('Botanist AI is typing...', style: TextStyle(fontSize: 11, color: Colors.grey)),
                  ],
                ),
              ),
            ),

          // Suggestion Chips
          SizedBox(
            height: 40,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 12),
              itemCount: suggestions.length,
              itemBuilder: (context, index) {
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4.0),
                  child: ActionChip(
                    label: Text(
                      suggestions[index],
                      style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w500),
                    ),
                    backgroundColor: colors.surface,
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                    onPressed: () => _sendMessage(suggestions[index]),
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 8),

          // Message Input Field
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: colors.surface,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 10,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            child: SafeArea(
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _messageController,
                      style: const TextStyle(fontSize: 14),
                      decoration: InputDecoration(
                        hintText: 'Ask a question about this diagnosis...',
                        hintStyle: const TextStyle(fontSize: 14, color: Colors.grey),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                        fillColor: isDark ? const Color(0xFF1B201D) : const Color(0xFFF0F4F2),
                        filled: true,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(24),
                          borderSide: BorderSide.none,
                        ),
                      ),
                      onSubmitted: _sendMessage,
                    ),
                  ),
                  const SizedBox(width: 8),
                  CircleAvatar(
                    backgroundColor: colors.primary,
                    child: IconButton(
                      icon: const Icon(Icons.send, color: Colors.white, size: 18),
                      onPressed: () => _sendMessage(_messageController.text),
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
