#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video.hpp>
#include "android/bitmap.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <__filesystem/directory_iterator.h>
#include <jni.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <android/log.h>


namespace fs = std::filesystem;
std::string gDatasetPath;
using namespace cv;
using namespace std;

void bitmapToMat(JNIEnv * env, jobject bitmap, cv::Mat &dst, jboolean
needUnPremultiplyAlpha){
    AndroidBitmapInfo info;
    void* pixels = 0;
    try {
        CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
        CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                   info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
        CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
        CV_Assert( pixels );
        dst.create(info.height, info.width, CV_8UC4);
        if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
        {
            cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if(needUnPremultiplyAlpha) cvtColor(tmp, dst, cv::COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, cv::COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch(const cv::Exception& e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        //jclass je = env->FindClass("org/opencv/core/CvException");
        jclass je = env->FindClass("java/lang/Exception");
        //if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}
void matToBitmap(JNIEnv * env, cv::Mat src, jobject bitmap, jboolean needPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void* pixels = 0;
    try {
        CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
        CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                   info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
        CV_Assert( src.dims == 2 && info.height == (uint32_t)src.rows && info.width ==
                                                                         (uint32_t)src.cols );
        CV_Assert( src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4 );
        CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
        CV_Assert( pixels );
        if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
        {
            cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if(src.type() == CV_8UC1)
            {
                cvtColor(src, tmp, cv::COLOR_GRAY2RGBA);
            } else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, cv::COLOR_RGB2RGBA);
            } else if(src.type() == CV_8UC4){
                if(needPremultiplyAlpha) cvtColor(src, tmp, cv::COLOR_RGBA2mRGBA);
                else src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if(src.type() == CV_8UC1)
            {
                cvtColor(src, tmp, cv::COLOR_GRAY2BGR565);
            } else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, cv::COLOR_RGB2BGR565);
            } else if(src.type() == CV_8UC4){
                cvtColor(src, tmp, cv::COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch(const cv::Exception& e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        //jclass je = env->FindClass("org/opencv/core/CvException");
        jclass je = env->FindClass("java/lang/Exception");
        //if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}

void logVector(const std::vector<double>& vec, const char* tag, const char* label) {
    std::string s = label;
    for (size_t i = 0; i < vec.size(); i++) {
        s += std::to_string(vec[i]) + " ";
    }
    __android_log_print(ANDROID_LOG_DEBUG, tag, "%s", s.c_str());
}

// Structure to load features from each figure
struct ShapeData {
    vector<double> huMoments;
    vector<double> zernikeMoments;
    string label;
};

// Hu Moments Calculation
vector<double> calculateHuMoments(Mat &image) {
    Moments moments = cv::moments(image, true);
    vector<double> huMoments(7);
    HuMoments(moments, huMoments);
    for (double &val : huMoments) {
        val= std::log1p(std::abs(val));
    }
    logVector(huMoments, "HuMoments", "Valores Hu: ");
    return huMoments;
}

// Factorial
double factorial(int n) {
    return (n == 0 || n == 1) ? 1 : n * factorial(n - 1);
}

// Zernike Radial Polynomial
double radialPolynomial(int n, int m, double r) {
    double sum = 0.0;
    int s_max = (n - abs(m)) / 2;
    for (int s = 0; s <= s_max; s++) {
        double numerator = pow(-1, s) * factorial(n - s);
        double denominator = factorial(s) *
                             factorial((n + abs(m)) / 2 - s) *
                             factorial((n - abs(m)) / 2 - s);
        sum += (numerator / denominator) * pow(r, n - 2 * s);
    }
    return sum;
}

// Zernike Moments Calculation (similar a Mahotas)
vector<double> calculateZernikeMoments(Mat &binaryImage) {
    int maxOrder = 4;
    int cx = binaryImage.cols / 2;
    int cy = binaryImage.rows / 2;
    double normFactor = sqrt(cx * cx + cy * cy);
    vector<double> moments;

    for (int n = 0; n <= maxOrder; n++) {
        for (int m = -n; m <= n; m += 2) {
            complex<double> moment(0.0, 0.0);

            for (int y = 0; y < binaryImage.rows; y++) {
                for (int x = 0; x < binaryImage.cols; x++) {
                    if (binaryImage.at<uchar>(y, x) > 0) {
                        double normX = (x - cx) / normFactor;
                        double normY = (cy - y) / normFactor; // (cy - y) mantiene la orientación
                        double r = sqrt(normX * normX + normY * normY);
                        double theta = atan2(normY, normX);

                        if (r <= 1.0) {
                            double R = radialPolynomial(n, m, r);
                            // Calcular el término Zernike: R * exp(i * m * theta)
                            complex<double> Z(R * cos(m * theta), R * sin(m * theta));
                            moment += Z * (binaryImage.at<uchar>(y, x) / 255.0);
                        }
                    }
                }
            }

            double factor = double(n + 1) / M_PI;
            double value = abs(moment * factor);
            moments.push_back(value);
        }
    }

    return moments;
}

std::vector<ShapeData> loadDatasetCSV(const std::string &csvFilePath) {
    std::vector<ShapeData> dataset;
    std::ifstream file(csvFilePath);
    if (!file.is_open()) {
        std::cerr << "Error al carga el CSV en: " << csvFilePath << std::endl;
        return dataset;
    }

    std::string line;
    // Leer y descartar el encabezado (si lo tiene)
    if (std::getline(file, line)) {
        // Aquí podrías procesar el header si es necesario.
    }
    // Number of Hu and Zernike Moments
    const int huCount = 7;
    const int zernikeCount = 10;

    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(lineStream, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < static_cast<size_t>(2 + huCount + zernikeCount)) {
            // No minimun colums
            continue;
        }
        // tokens[1] category (label)
        std::string label = tokens[1];

        // Extracting Hu moments: from tokens[2] to tokens[2 + huCount - 1]
        std::vector<double> huMoments;
        for (int i = 0; i < huCount; ++i) {
            try {
                huMoments.push_back(std::stod(tokens[2 + i]));
            } catch (const std::exception &e) {
                huMoments.push_back(0.0);
            }
        }

        // Extracting Zernike moments: from tokens[2 + huCount] to tokens[2 + huCount + zernikeCount - 1]
        std::vector<double> zernikeMoments;
        for (int i = 0; i < zernikeCount; ++i) {
            try {
                zernikeMoments.push_back(std::stod(tokens[2 + huCount + i]));
            } catch (const std::exception &e) {
                zernikeMoments.push_back(0.0);
            }
        }

        dataset.push_back({huMoments, zernikeMoments, label});
    }

    file.close();
    return dataset;
}

//  Classify the image using Hu or Zernike
string classifyShape(const vector<double> &features, const vector<ShapeData> &dataset, bool useHu) {
    double minDistance = DBL_MAX;
    string bestMatch = "Desconocido";

    for (const auto &data : dataset) {
        double distance = 0.0;
        const vector<double> &comparisonFeatures = useHu ? data.huMoments : data.zernikeMoments;

        for (size_t i = 0; i < features.size(); i++) {
            distance += pow(features[i] - comparisonFeatures[i], 2);
        }

        distance = sqrt(distance);
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = data.label;
        }
    }

    return bestMatch;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_hu_1zernike_1classifier_MainActivity_classifyShapeHu(
        JNIEnv *env,
        jobject thiz,
        jobject bitmap) {
    Mat src;
    bitmapToMat(env, bitmap, src, false);

    //  Preprocessing
    Mat gray, binary;
    cvtColor(src, gray, COLOR_RGBA2GRAY);
    threshold(gray, binary, 128, 255, THRESH_BINARY_INV);

    // HU Features
    vector<double> features = calculateHuMoments(binary);

    // (Opcional) Imprimir también el vector de características calculado
    logVector(features, "HuClassifier", "Features calculadas: ");

    // Load Dataset Moments
    vector<ShapeData> dataset = loadDatasetCSV(gDatasetPath);

    // Classify using HU
    string result = classifyShape(features, dataset, false);

    return env->NewStringUTF(result.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_hu_1zernike_1classifier_MainActivity_classifyShapeZernike(
        JNIEnv *env,
        jobject thiz,
        jobject bitmap) {
    Mat src;
    bitmapToMat(env, bitmap, src, false);

    // Preprocessing
    Mat gray, binary;
    cvtColor(src, gray, COLOR_RGBA2GRAY);
    threshold(gray, binary, 128, 255, THRESH_BINARY_INV);

    // Zernike Features
    vector<double> features = calculateZernikeMoments(binary);

    // (Opcional) Imprimir también el vector de características calculado
    logVector(features, "ZernikeClassifier", "Features calculadas: ");

    // Load Dataset Moments
    vector<ShapeData> dataset = loadDatasetCSV(gDatasetPath);

    // Classify using Zernike
    string result = classifyShape(features, dataset, false);

    return env->NewStringUTF(result.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_hu_1zernike_1classifier_MainActivity_setDatasetPath(JNIEnv *env, jobject thiz,
                                                                     jstring path) {
    const char *nativePath = env->GetStringUTFChars(path, 0);
    gDatasetPath = nativePath;
    env->ReleaseStringUTFChars(path, nativePath);
}