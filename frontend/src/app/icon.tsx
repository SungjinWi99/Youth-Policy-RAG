import { ImageResponse } from "next/og";

export const size = {
  width: 64,
  height: 64,
};

export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "white",
          background: "#1d56bc",
          borderRadius: 16,
          fontSize: 24,
          fontWeight: 800,
          letterSpacing: -1,
        }}
      >
        YP
      </div>
    ),
    size,
  );
}
