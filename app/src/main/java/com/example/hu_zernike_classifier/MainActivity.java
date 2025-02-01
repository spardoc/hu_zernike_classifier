package com.example.hu_zernike_classifier;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Adapter;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import com.example.hu_zernike_classifier.databinding.ActivityMainBinding;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'hu_zernike_classifier' library on application startup.
    static {
        System.loadLibrary("hu_zernike_classifier");
    }

    private ActivityMainBinding binding;
    private Spinner spinner;
    private ArrayAdapter<String> adapter;
    private String[] items;
    private TextView result;
    private Button classifier_button;
    private Button clear_button;
    private DrawingView drawingView;
    private Bitmap drawnBitmap;
    private String selectedMethod = "Hu";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        String destPath = getFilesDir() + "/moments_dataset.csv";
        copyAssetToInternalStorage("moments_dataset.csv", destPath);
        setDatasetPath(destPath);


        result = findViewById(R.id.result_label);
        classifier_button = findViewById(R.id.classifier_button);
        clear_button = findViewById(R.id.clear_button);
        spinner = findViewById(R.id.spinner);
        drawingView = findViewById(R.id.drawing_view);
        items = getResources().getStringArray(R.array.spinner_items);

        adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, items);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                selectedMethod = parent.getItemAtPosition(position).toString();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        AssetManager assetManager = getAssets();
        InputStream inputStream = null;
        Bitmap bitmap = BitmapFactory.decodeStream(inputStream);

        classifier_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawnBitmap = drawingView.getDrawingBitmap();
                if (drawnBitmap != null) {
                    int targetWidth = 150;
                    int targetHeight = 150;
                    drawnBitmap = resizeBitmap(drawnBitmap, targetWidth, targetHeight);
                    String classificationResult;
                    if (selectedMethod.equals("Momentos de Hu")) {
                        classificationResult = classifyShapeHu(drawnBitmap);
                    } else {
                        classificationResult = classifyShapeZernike(drawnBitmap);
                    }
                    result.setText(classificationResult);
                } else {
                    result.setText("No se ha dibujado ninguna imagen");
                }
            }
        });

        clear_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawingView.clearDrawing();
            }
        });

    }

    public void copyAssetToInternalStorage(String assetName, String destinationPath) {
        AssetManager assetManager = getAssets();
        InputStream in = null;
        FileOutputStream out = null;
        try {
            in = assetManager.open(assetName);
            File outFile = new File(destinationPath);
            out = new FileOutputStream(outFile);

            byte[] buffer = new byte[1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            out.flush();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (in != null) in.close();
                if (out != null) out.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // Función para redimensionar la imagen manteniendo la relación de aspecto
    private Bitmap resizeBitmap(Bitmap original, int targetWidth, int targetHeight) {
        int width = original.getWidth();
        int height = original.getHeight();

        // Mantener la relación de aspecto de la imagen original
        float aspectRatio = (float) width / height;
        if (width > height) {
            targetHeight = (int) (targetWidth / aspectRatio);
        } else {
            targetWidth = (int) (targetHeight * aspectRatio);
        }

        return Bitmap.createScaledBitmap(original, targetWidth, targetHeight, false);
    }


    public native String classifyShapeHu(Bitmap bitmap);
    public native String classifyShapeZernike(Bitmap bitmap);
    public native void setDatasetPath(String path);

}