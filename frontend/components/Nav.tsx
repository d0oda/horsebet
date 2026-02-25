"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
    { href: "/", label: "Dashboard" },
    { href: "/backtest", label: "Backtest" },
    { href: "/bankroll", label: "Bankroll" },
    { href: "/settings", label: "Settings" },
];

export default function Nav() {
    const pathname = usePathname();

    return (
        <nav className="nav-bar">
            <div className="nav-inner">
                <Link href="/" className="nav-logo">
                    <span className="logo-icon">⚡</span>
                    <span className="logo-text">UmaEdge</span>
                </Link>

                <div className="nav-links">
                    {links.map((link) => (
                        <Link
                            key={link.href}
                            href={link.href}
                            className={`nav-link ${pathname === link.href ? "active" : ""}`}
                        >
                            {link.label}
                        </Link>
                    ))}
                </div>

                <div className="nav-right">
                    <div className="nav-avatar">R</div>
                </div>
            </div>

            <style jsx>{`
        .nav-bar {
          background: rgba(10, 13, 20, 0.85);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          border-bottom: 1px solid rgba(42, 48, 64, 0.6);
          position: sticky;
          top: 0;
          z-index: 100;
        }
        .nav-inner {
          max-width: 1400px;
          margin: 0 auto;
          padding: 0 2rem;
          height: 60px;
          display: flex;
          align-items: center;
          gap: 2rem;
        }
        .nav-logo {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          text-decoration: none;
        }
        .logo-icon {
          font-size: 1.4rem;
        }
        .logo-text {
          font-size: 1.25rem;
          font-weight: 800;
          color: #00ff88;
          letter-spacing: -0.03em;
        }
        .nav-links {
          display: flex;
          gap: 0.25rem;
          flex: 1;
        }
        .nav-link {
          padding: 0.5rem 1rem;
          border-radius: 6px;
          font-size: 0.875rem;
          font-weight: 500;
          color: #8b95a5;
          text-decoration: none;
          transition: all 0.2s;
        }
        .nav-link:hover {
          color: #f0f0f0;
          background: rgba(42, 48, 64, 0.5);
        }
        .nav-link.active {
          color: #00ff88;
          background: rgba(0, 255, 136, 0.1);
        }
        .nav-right {
          display: flex;
          align-items: center;
        }
        .nav-avatar {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: linear-gradient(135deg, #00ff88, #4da6ff);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 0.8rem;
          font-weight: 700;
          color: #0a0d14;
        }
      `}</style>
        </nav>
    );
}
