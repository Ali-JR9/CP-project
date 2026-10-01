#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

using namespace std;

const string EMBED_MODEL = "gemini-embedding-001";
const int DIMENSIONS = 768;
const int CHUNK_SIZE = 1200;
const int TOP_PARTS = 5;

struct Topic {
    vector<string> keywords;
    string answer;
};

struct Chunk {
    string text;
    vector<float> vec;
};

void waitSeconds(int seconds) {
#ifdef _WIN32
    Sleep(seconds * 1000);
#else
    sleep(seconds);
#endif
}

string numberToText(int number) {
    ostringstream stream;
    stream << number;
    return stream.str();
}

string toLowerCase(string text) {
    transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return tolower(c);
    });
    return text;
}

string trim(string text) {
    size_t start = text.find_first_not_of(" \t\r\n");
    if (start == string::npos) {
        return "";
    }
    size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(start, end - start + 1);
}

string fixNewLines(string text) {
    size_t pos = text.find("\\n");
    while (pos != string::npos) {
        text.replace(pos, 2, "\n");
        pos = text.find("\\n", pos + 1);
    }
    return text;
}

string readWholeFile(string filename) {
    ifstream file(filename.c_str(), ios::binary);
    if (!file.is_open()) {
        return "";
    }
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

string jsonEscape(string text) {
    string result = "";

    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = text[i];

        if (c == '"') {
            result += "\\\"";
        } else if (c == '\\') {
            result += "\\\\";
        } else if (c == '\n') {
            result += "\\n";
        } else if (c == '\r') {
            result += "\\r";
        } else if (c == '\t') {
            result += "\\t";
        } else if (c < 32) {
            char buffer[8];
            sprintf(buffer, "\\u%04x", c);
            result += buffer;
        } else {
            result += (char)c;
        }
    }

    return result;
}

unsigned int hexValue(string text, size_t pos) {
    if (pos + 4 > text.size()) {
        return 0;
    }
    return (unsigned int)strtoul(text.substr(pos, 4).c_str(), NULL, 16);
}

void appendUtf8(string& out, unsigned int code) {
    if (code < 0x80) {
        out += (char)code;
    } else if (code < 0x800) {
        out += (char)(0xC0 | (code >> 6));
        out += (char)(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
        out += (char)(0xE0 | (code >> 12));
        out += (char)(0x80 | ((code >> 6) & 0x3F));
        out += (char)(0x80 | (code & 0x3F));
    } else {
        out += (char)(0xF0 | (code >> 18));
        out += (char)(0x80 | ((code >> 12) & 0x3F));
        out += (char)(0x80 | ((code >> 6) & 0x3F));
        out += (char)(0x80 | (code & 0x3F));
    }
}

string readAllStrings(string response, string key) {
    string search = "\"" + key + "\"";
    string result = "";
    size_t pos = response.find(search);

    while (pos != string::npos) {
        size_t i = pos + search.size();
        size_t next = i;

        while (i < response.size() && isspace((unsigned char)response[i])) {
            i++;
        }

        if (i < response.size() && response[i] == ':') {
            i++;
            while (i < response.size() && isspace((unsigned char)response[i])) {
                i++;
            }

            if (i < response.size() && response[i] == '"') {
                i++;

                while (i < response.size() && response[i] != '"') {
                    char c = response[i];

                    if (c == '\\' && i + 1 < response.size()) {
                        char letter = response[i + 1];
                        i += 2;

                        if (letter == 'n') {
                            result += '\n';
                        } else if (letter == 't') {
                            result += '\t';
                        } else if (letter == 'r') {
                            result += "";
                        } else if (letter == 'u') {
                            unsigned int code = hexValue(response, i);
                            i += 4;

                            if (code >= 0xD800 && code <= 0xDBFF && i + 6 <= response.size() && response[i] == '\\' && response[i + 1] == 'u') {
                                unsigned int low = hexValue(response, i + 2);
                                i += 6;
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                            }

                            appendUtf8(result, code);
                        } else {
                            result += letter;
                        }
                    } else {
                        result += c;
                        i++;
                    }
                }

                next = i;
            }
        }

        pos = response.find(search, next);
    }

    return result;
}

int readNumber(string response, string key) {
    size_t pos = response.find("\"" + key + "\"");
    if (pos == string::npos) {
        return 0;
    }
    size_t colon = response.find(':', pos);
    if (colon == string::npos) {
        return 0;
    }
    return atoi(response.c_str() + colon + 1);
}

vector<vector<float> > parseEmbeddings(string response) {
    vector<vector<float> > result;
    size_t pos = response.find("\"values\"");

    while (pos != string::npos) {
        size_t open = response.find('[', pos);
        if (open == string::npos) {
            break;
        }
        size_t close = response.find(']', open);
        if (close == string::npos) {
            break;
        }

        vector<float> vec;
        const char* p = response.c_str() + open + 1;
        const char* limit = response.c_str() + close;

        while (p < limit) {
            char* next;
            double value = strtod(p, &next);
            if (next == p) {
                p++;
            } else {
                vec.push_back((float)value);
                p = next;
            }
        }

        result.push_back(vec);
        pos = response.find("\"values\"", close);
    }

    return result;
}

vector<string> splitKeywords(string text) {
    vector<string> words;
    string current = "";

    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ',') {
            string word = toLowerCase(trim(current));
            if (word != "") {
                words.push_back(word);
            }
            current = "";
        } else {
            current += text[i];
        }
    }

    string word = toLowerCase(trim(current));
    if (word != "") {
        words.push_back(word);
    }

    return words;
}

bool loadFaq(string filename, vector<Topic>& database) {
    ifstream file(filename.c_str());

    if (!file.is_open()) {
        return false;
    }

    string line;
    while (getline(file, line)) {
        line = trim(line);

        if (line == "" || line[0] == '#') {
            continue;
        }

        size_t divider = line.find('|');
        if (divider == string::npos) {
            continue;
        }

        Topic topic;
        topic.keywords = splitKeywords(line.substr(0, divider));
        topic.answer = fixNewLines(trim(line.substr(divider + 1)));

        if (!topic.keywords.empty() && topic.answer != "") {
            database.push_back(topic);
        }
    }

    return true;
}

int getScore(string query, Topic topic) {
    int score = 0;
    for (size_t i = 0; i < topic.keywords.size(); i++) {
        if (query.find(topic.keywords[i]) != string::npos) {
            score++;
        }
    }
    return score;
}

string searchFaq(string query, vector<Topic>& database) {
    int bestScore = 0;
    int bestIndex = -1;

    for (int i = 0; i < (int)database.size(); i++) {
        int score = getScore(query, database[i]);
        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }

    if (bestIndex == -1) {
        return "";
    }
    return database[bestIndex].answer;
}

void loadSettings(string& apiKey, string& model) {
    ifstream file("apikey.txt");

    if (!file.is_open()) {
        return;
    }

    string line;
    if (getline(file, line)) {
        apiKey = trim(line);
    }
    if (getline(file, line)) {
        string newModel = trim(line);
        if (newModel != "") {
            model = newModel;
        }
    }
}

void normalize(vector<float>& vec) {
    float sum = 0;
    for (size_t i = 0; i < vec.size(); i++) {
        sum += vec[i] * vec[i];
    }

    float length = sqrt(sum);
    if (length > 0) {
        for (size_t i = 0; i < vec.size(); i++) {
            vec[i] = vec[i] / length;
        }
    }
}

vector<string> splitIntoChunks(string text) {
    vector<string> chunks;
    stringstream stream(text);
    string line;
    string current = "";

    while (getline(stream, line)) {
        line = trim(line);

        if (line == "") {
            continue;
        }

        current += line + "\n";

        if ((int)current.size() >= CHUNK_SIZE) {
            chunks.push_back(current);
            current = line + "\n";
        }
    }

    if (trim(current) != "") {
        chunks.push_back(current);
    }

    return chunks;
}

string callApi(string url, string apiKey, string payload) {
    for (int attempt = 1; attempt <= 5; attempt++) {
        ofstream out("request.json", ios::binary);
        out << payload;
        out.close();

        remove("response.json");

        string command = "curl -s --max-time 120 -X POST -H \"Content-Type: application/json\" -H \"x-goog-api-key: " + apiKey + "\" --data-binary @request.json -o response.json \"" + url + "\"";
        int result = system(command.c_str());

        string response = readWholeFile("response.json");
        remove("request.json");
        remove("response.json");

        if (result != 0 || response == "") {
            cout << "\n(Could not reach Google. Check your internet, and check that 'curl --version' works in Command Prompt.)\n";
            return "";
        }

        size_t errorPos = response.find("\"error\"");

        if (errorPos == string::npos || errorPos > 10) {
            return response;
        }

        int status = readNumber(response, "code");

        if ((status == 429 || status >= 500) && attempt < 5) {
            cout << "  (Google is busy, waiting " << 20 * attempt << " seconds...)\n";
            waitSeconds(20 * attempt);
        } else {
            cout << "\n(Google error, status code: " << status << ")\n";
            cout << readAllStrings(response, "message") << "\n";
            return "";
        }
    }

    return "";
}

bool embedBatch(string apiKey, vector<string> texts, string taskType, vector<vector<float> >& results) {
    string url = "https://generativelanguage.googleapis.com/v1beta/models/" + EMBED_MODEL + ":batchEmbedContents";

    string payload = "{\"requests\":[";
    for (int i = 0; i < (int)texts.size(); i++) {
        if (i > 0) {
            payload += ",";
        }
        payload += "{\"model\":\"models/" + EMBED_MODEL + "\",";
        payload += "\"content\":{\"parts\":[{\"text\":\"" + jsonEscape(texts[i]) + "\"}]},";
        payload += "\"taskType\":\"" + taskType + "\",";
        payload += "\"outputDimensionality\":" + numberToText(DIMENSIONS) + "}";
    }
    payload += "]}";

    string response = callApi(url, apiKey, payload);

    if (response == "") {
        return false;
    }

    vector<vector<float> > vectors = parseEmbeddings(response);

    if (vectors.size() != texts.size()) {
        return false;
    }

    for (size_t i = 0; i < vectors.size(); i++) {
        normalize(vectors[i]);
        results.push_back(vectors[i]);
    }

    return true;
}

bool saveIndex(vector<Chunk>& chunks, size_t prospectusSize) {
    ofstream file("index.txt", ios::binary);

    if (!file.is_open()) {
        return false;
    }

    file << chunks.size() << " " << prospectusSize << "\n";

    for (size_t i = 0; i < chunks.size(); i++) {
        file << chunks[i].text.size() << "\n";
        file << chunks[i].text << "\n";
        file << chunks[i].vec.size();

        for (size_t j = 0; j < chunks[i].vec.size(); j++) {
            file << " " << chunks[i].vec[j];
        }

        file << "\n";
    }

    return true;
}

bool loadIndex(vector<Chunk>& chunks, size_t prospectusSize) {
    ifstream file("index.txt", ios::binary);

    if (!file.is_open()) {
        return false;
    }

    size_t count = 0;
    size_t savedSize = 0;
    file >> count >> savedSize;

    if (!file || savedSize != prospectusSize) {
        return false;
    }

    for (size_t i = 0; i < count; i++) {
        size_t length = 0;
        file >> length;
        file.ignore(1);

        Chunk chunk;
        chunk.text = string(length, ' ');
        file.read(&chunk.text[0], length);
        file.ignore(1);

        size_t size = 0;
        file >> size;
        chunk.vec = vector<float>(size);

        for (size_t j = 0; j < size; j++) {
            file >> chunk.vec[j];
        }

        if (!file) {
            chunks.clear();
            return false;
        }

        chunks.push_back(chunk);
    }

    return !chunks.empty();
}

bool buildIndex(string apiKey, string prospectus, vector<Chunk>& chunks) {
    vector<string> pieces = splitIntoChunks(prospectus);
    int batchSize = 50;

    cout << "Prospectus is split into " << pieces.size() << " parts.\n";

    for (int start = 0; start < (int)pieces.size(); start += batchSize) {
        int end = min(start + batchSize, (int)pieces.size());
        vector<string> batch(pieces.begin() + start, pieces.begin() + end);
        vector<vector<float> > vectors;

        if (!embedBatch(apiKey, batch, "RETRIEVAL_DOCUMENT", vectors)) {
            chunks.clear();
            return false;
        }

        for (int i = 0; i < (int)batch.size(); i++) {
            Chunk chunk;
            chunk.text = batch[i];
            chunk.vec = vectors[i];
            chunks.push_back(chunk);
        }

        cout << "Done " << end << " of " << pieces.size() << "\n";
        waitSeconds(3);
    }

    return saveIndex(chunks, prospectus.size());
}

vector<int> findRelevant(vector<float> queryVec, vector<Chunk>& chunks, int count) {
    vector<pair<float, int> > scores;

    for (int i = 0; i < (int)chunks.size(); i++) {
        float score = 0;
        size_t size = min(queryVec.size(), chunks[i].vec.size());

        for (size_t j = 0; j < size; j++) {
            score += queryVec[j] * chunks[i].vec[j];
        }

        scores.push_back(make_pair(score, i));
    }

    sort(scores.begin(), scores.end(), [](pair<float, int> a, pair<float, int> b) {
        return a.first > b.first;
    });

    vector<int> best;
    for (int i = 0; i < count && i < (int)scores.size(); i++) {
        best.push_back(scores[i].second);
    }

    return best;
}

string askGemini(string apiKey, string model, string context, string question) {
    string url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent";

    string rules = "You are the Digital Assistant of Mehran University of Engineering and Technology (MUET). "
                   "Answer the student's question using ONLY the prospectus extracts provided. "
                   "If the answer is not in the extracts, say that the information was not found "
                   "and ask the student to contact the university office. "
                   "Keep the answer short and clear. Reply in the same language the student used.";

    string message = "PROSPECTUS EXTRACTS:\n" + context + "\n\nSTUDENT QUESTION:\n" + question;

    string payload = "{\"system_instruction\":{\"parts\":[{\"text\":\"" + jsonEscape(rules) + "\"}]},";
    payload += "\"contents\":[{\"parts\":[{\"text\":\"" + jsonEscape(message) + "\"}]}]}";

    string response = callApi(url, apiKey, payload);

    if (response == "") {
        return "";
    }

    return trim(readAllStrings(response, "text"));
}

string normalizeQuestion(string text) {
    string result = "";
    bool lastSpace = true;

    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = text[i];

        if (c >= 128 || isalnum(c)) {
            result += (char)tolower(c);
            lastSpace = false;
        } else if (!lastSpace) {
            result += ' ';
            lastSpace = true;
        }
    }

    return trim(result);
}

void loadCache(unordered_map<string, string>& cache) {
    ifstream file("cache.txt", ios::binary);

    if (!file.is_open()) {
        return;
    }

    string key;
    while (getline(file, key)) {
        size_t length = 0;
        file >> length;
        file.ignore(1);

        if (!file || length > 1000000) {
            break;
        }

        string answer(length, ' ');
        file.read(&answer[0], length);
        file.ignore(1);

        if (!file) {
            break;
        }

        cache[key] = answer;
    }
}

void saveToCache(string key, string answer) {
    ofstream file("cache.txt", ios::binary | ios::app);

    if (file.is_open()) {
        file << key << "\n" << answer.size() << "\n" << answer << "\n";
    }
}

int main() {
#ifdef _WIN32
    system("chcp 65001 > nul");
#endif

    string apiKey = "";
    string model = "gemini-3.5-flash";
    loadSettings(apiKey, model);

    bool keyReady = (apiKey != "" && apiKey != "PASTE_YOUR_GEMINI_API_KEY_HERE");

    vector<Topic> faqDatabase;
    bool faqLoaded = loadFaq("faq.txt", faqDatabase);

    vector<Chunk> chunks;
    unordered_map<string, string> cache;
    bool aiReady = false;

    if (keyReady) {
        string prospectus = readWholeFile("prospectus.txt");

        if (trim(prospectus) == "") {
            cout << "prospectus.txt is empty or missing.\n";
        } else if (loadIndex(chunks, prospectus.size())) {
            aiReady = true;
        } else {
            remove("cache.txt");
            cout << "Building the search index. This happens only once and may take a few minutes.\n";
            aiReady = buildIndex(apiKey, prospectus, chunks);

            if (!aiReady) {
                cout << "Index could not be built. Check your API key and internet, then run again.\n";
            }
        }
    } else {
        cout << "Gemini API key is missing in apikey.txt.\n";
    }

    loadCache(cache);

    cout << "\n=======================================\n";
    cout << "     WELCOME TO DIGITAL ASSISTANT      \n";
    cout << "=======================================\n";

    if (aiReady) {
        cout << "Mode: AI answers from the MUET prospectus (" << chunks.size() << " parts loaded)\n";
    } else {
        cout << "Mode: keyword answers from faq.txt only\n";
    }

    if (!faqLoaded) {
        cout << "Note: faq.txt was not found, so there is no backup database.\n";
    }

    cout << "Saved answers: " << cache.size() << "\n";
    cout << "Type 'exit' to quit (or 'clear cache' to erase saved answers).\n\n";

    string userQuery;

    while (true) {
        cout << "Ask a question : ";
        getline(cin, userQuery);

        string question = trim(userQuery);

        if (toLowerCase(question) == "exit") {
            cout << "Thank you for using the Digital Assistant. Goodbye!\n";
            break;
        }

        if (question.empty()) {
            continue;
        }

        if (toLowerCase(question) == "clear cache") {
            cache.clear();
            remove("cache.txt");
            cout << "Saved answers erased.\n\n";
            continue;
        }

        string key = normalizeQuestion(question);
        string answer = "";

        if (key != "" && cache.find(key) != cache.end()) {
            answer = cache[key];
        } else if (aiReady) {
            vector<string> oneQuestion;
            oneQuestion.push_back(question);
            vector<vector<float> > questionVec;

            if (embedBatch(apiKey, oneQuestion, "RETRIEVAL_QUERY", questionVec) && !questionVec.empty()) {
                vector<int> best = findRelevant(questionVec[0], chunks, TOP_PARTS);
                string context = "";

                for (size_t i = 0; i < best.size(); i++) {
                    context += "--- Extract ---\n" + chunks[best[i]].text + "\n";
                }

                answer = askGemini(apiKey, model, context, question);

                if (answer != "" && key != "") {
                    cache[key] = answer;
                    saveToCache(key, answer);
                }
            }
        }

        if (answer == "" && faqLoaded) {
            answer = searchFaq(toLowerCase(question), faqDatabase);
        }

        if (answer == "") {
            cout << "\n[ANSWER]: Sorry, no information found for that query.\n";
            cout << "          Please visit the Main Administration Desk in Block A.\n\n";
        } else {
            cout << "\n[ANSWER]: " << answer << "\n\n";
        }
    }

    return 0;
}
