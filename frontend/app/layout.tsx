import type { Metadata } from "next";
import Script from "next/script";

export const metadata: Metadata = {
  title: "UmaEdge Raceday Dashboard",
  description: "AI-powered Japanese horse racing intelligence platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <script dangerouslySetInnerHTML={{ __html: `
          tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    "colors": {
                        "on-error-container": "#ffdad6",
                        "on-secondary": "#00344d",
                        "surface-dim": "#131315",
                        "on-tertiary-container": "#004139",
                        "on-tertiary": "#003731",
                        "primary-fixed-dim": "#4edea3",
                        "on-background": "#e5e1e4",
                        "on-secondary-container": "#00344e",
                        "on-error": "#690005",
                        "surface-container": "#201f22",
                        "secondary": "#89ceff",
                        "error-container": "#93000a",
                        "on-primary-fixed": "#002113",
                        "primary": "#4edea3",
                        "tertiary-container": "#0db6a4",
                        "tertiary-fixed": "#71f8e4",
                        "secondary-fixed": "#c9e6ff",
                        "secondary-container": "#00a2e6",
                        "surface-container-low": "#1c1b1d",
                        "on-primary-fixed-variant": "#005236",
                        "outline": "#86948a",
                        "on-tertiary-fixed-variant": "#005048",
                        "secondary-fixed-dim": "#89ceff",
                        "error": "#ffb4ab",
                        "inverse-surface": "#e5e1e4",
                        "on-primary-container": "#00422b",
                        "on-primary": "#003824",
                        "background": "#131315",
                        "outline-variant": "#3c4a42",
                        "on-secondary-fixed-variant": "#004c6e",
                        "on-surface": "#e5e1e4",
                        "surface-variant": "#353437",
                        "surface-bright": "#39393b",
                        "surface-container-highest": "#353437",
                        "tertiary": "#4fdbc8",
                        "inverse-primary": "#006c49",
                        "surface-container-high": "#2a2a2c",
                        "tertiary-fixed-dim": "#4fdbc8",
                        "on-secondary-fixed": "#001e2f",
                        "on-surface-variant": "#bbcabf",
                        "primary-fixed": "#6ffbbe",
                        "surface-tint": "#4edea3",
                        "inverse-on-surface": "#313032",
                        "surface-container-lowest": "#0e0e10",
                        "primary-container": "#10b981",
                        "surface": "#131315",
                        "on-tertiary-fixed": "#00201c"
                    },
                    "borderRadius": {
                        "DEFAULT": "0.125rem",
                        "lg": "0.25rem",
                        "xl": "0.5rem",
                        "full": "0.75rem"
                    },
                    "spacing": {
                        "gutter": "16px",
                        "sm": "8px",
                        "xl": "32px",
                        "xs": "4px",
                        "base": "4px",
                        "container-margin": "24px",
                        "md": "16px",
                        "lg": "24px"
                    },
                    "fontFamily": {
                        "title-md": ["Inter"],
                        "display-lg": ["Inter"],
                        "headline-lg": ["Inter"],
                        "body-md": ["Inter"],
                        "body-sm": ["Inter"],
                        "data-table": ["Inter"],
                        "label-caps": ["Inter"]
                    },
                    "fontSize": {
                        "title-md": ["18px", {"lineHeight": "1.4", "fontWeight": "600"}],
                        "display-lg": ["48px", {"lineHeight": "1.1", "letterSpacing": "-0.02em", "fontWeight": "700"}],
                        "headline-lg": ["32px", {"lineHeight": "1.2", "letterSpacing": "-0.01em", "fontWeight": "600"}],
                        "body-md": ["14px", {"lineHeight": "1.5", "fontWeight": "400"}],
                        "body-sm": ["13px", {"lineHeight": "1.5", "fontWeight": "400"}],
                        "data-table": ["13px", {"lineHeight": "1", "fontWeight": "500"}],
                        "label-caps": ["11px", {"lineHeight": "1", "letterSpacing": "0.05em", "fontWeight": "700"}]
                    }
                }
            }
          }
        ` }}></script>
        <style>{`
          .material-symbols-outlined {
              font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
          }
          /* Custom scrollbar for high-density data */
          ::-webkit-scrollbar { width: 6px; height: 6px; }
          ::-webkit-scrollbar-track { background: #131315; }
          ::-webkit-scrollbar-thumb { background: #3c4a42; border-radius: 3px; }
          ::-webkit-scrollbar-thumb:hover { background: #4edea3; }
          
          .tabular-nums { font-variant-numeric: tabular-nums; }
          
          /* Subtle glow for elevated interaction */
          .interaction-glow:focus-within {
              box-shadow: 0 0 0 2px rgba(78, 222, 163, 0.1);
          }
        `}</style>
      </head>
      <body className="bg-[#131315] text-[#e5e1e4] overflow-hidden flex h-screen m-0 p-0" style={{fontFamily: "'Inter', sans-serif"}}>
        {children}
      </body>
    </html>
  );
}
