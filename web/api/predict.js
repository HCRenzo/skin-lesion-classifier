// Función serverless de Vercel: hace de proxy hacia el endpoint de
// Databricks Model Serving. El token de Databricks vive acá (variables de
// entorno del proyecto en Vercel, nunca llega al navegador) -- el cliente
// solo le manda la imagen en base64 a esta función.
//
// Variables de entorno requeridas en el proyecto de Vercel:
//   DATABRICKS_HOST            ej. https://dbc-xxxx.cloud.databricks.com
//   DATABRICKS_TOKEN           el PAT generado en Databricks
//   SERVING_ENDPOINT_NAME      opcional, default "skin-lesion-classifier"

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const { image_b64 } = req.body || {};
  if (!image_b64) {
    res.status(400).json({ error: "Falta image_b64 en el body" });
    return;
  }

  const host = process.env.DATABRICKS_HOST;
  const token = process.env.DATABRICKS_TOKEN;
  const endpointName = process.env.SERVING_ENDPOINT_NAME || "skin-lesion-classifier";

  if (!host || !token) {
    res.status(500).json({
      error: "El servidor no tiene configuradas DATABRICKS_HOST / DATABRICKS_TOKEN (variables de entorno del proyecto en Vercel).",
    });
    return;
  }

  try {
    const url = `${host.replace(/\/$/, "")}/serving-endpoints/${endpointName}/invocations`;
    const dbResp = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataframe_split: { columns: ["image_b64"], data: [[image_b64]] },
      }),
    });

    if (!dbResp.ok) {
      const text = await dbResp.text();
      res.status(502).json({
        error: `El endpoint de Databricks respondió ${dbResp.status}`,
        detail: text.slice(0, 800),
      });
      return;
    }

    const data = await dbResp.json();
    const prediction = data.predictions && data.predictions[0];
    if (!prediction) {
      res.status(502).json({
        error: "Respuesta inesperada del endpoint (sin 'predictions')",
        detail: JSON.stringify(data).slice(0, 800),
      });
      return;
    }

    res.status(200).json(prediction);
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
};
