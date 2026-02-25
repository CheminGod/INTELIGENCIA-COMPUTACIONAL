package com.spam;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxMap;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class App {

    public static void main(String[] args) throws Exception {

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session = env.createSession("spam_model.onnx");

        String[] inputText = new String[]{
                "Congratulations! You have won a free prize. Click here now!"
        };

        OnnxTensor textTensor = OnnxTensor.createTensor(env, inputText);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        String inputName = session.getInputNames().iterator().next();
        inputs.put(inputName, textTensor);

        OrtSession.Result results = session.run(inputs);

        long[] label = (long[]) results.get("output_label").get().getValue();

        Object rawProb = results.get("output_probability").get().getValue();

        @SuppressWarnings("unchecked")
        List<?> probList = (List<?>) rawProb;

        OnnxMap onnxMap = (OnnxMap) probList.get(0);

        Map<?, ?> javaMap = onnxMap.getValue();

        float probNoSpam = ((Number) javaMap.get(0L)).floatValue();
        float probSpam = ((Number) javaMap.get(1L)).floatValue();

        System.out.println("Label predicho: " + label[0]);
        System.out.println("Probabilidad NO spam: " + probNoSpam);
        System.out.println("Probabilidad SPAM: " + probSpam);

        if (probSpam > 0.5) {
            System.out.println("El correo es SPAM");
        } else {
            System.out.println("El correo NO es spam");
        }

        textTensor.close();
        session.close();
        env.close();
    }
}