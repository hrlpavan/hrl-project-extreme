/**
 * HRL International Private Limited™ — Frontend Security & Code Protection Shield
 * Proprietary Asset Protection & Anti-Inspection Architecture
 * Founder: Pavan Kumar Sadashiv
 */

(function () {
  'use strict';

  // 1. Disable Right-Click Context Menu (Inspect / View Source Block)
  document.addEventListener('contextmenu', function (e) {
    // Allow context menu only inside legitimate input fields
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) {
      return;
    }
    e.preventDefault();
    return false;
  }, { capture: true });

  // 2. Intercept DevTools Keyboard Shortcuts (Windows, Linux, macOS)
  document.addEventListener('keydown', function (e) {
    // F12 key (Windows/Linux standard DevTools)
    if (e.key === 'F12' || e.keyCode === 123) {
      e.preventDefault();
      e.stopPropagation();
      return false;
    }

    const isCtrlOrCmd = e.ctrlKey || e.metaKey;

    if (isCtrlOrCmd) {
      // Ctrl+Shift+I (Windows/Linux Inspect) / Cmd+Opt+I (macOS Inspect)
      if (e.shiftKey && (e.key === 'I' || e.key === 'i' || e.keyCode === 73)) {
        e.preventDefault();
        e.stopPropagation();
        return false;
      }
      // Ctrl+Shift+J (Console) / Cmd+Opt+J (Console)
      if (e.shiftKey && (e.key === 'J' || e.key === 'j' || e.keyCode === 74)) {
        e.preventDefault();
        e.stopPropagation();
        return false;
      }
      // Ctrl+Shift+C (Element Inspector) / Cmd+Opt+C
      if (e.shiftKey && (e.key === 'C' || e.key === 'c' || e.keyCode === 67)) {
        e.preventDefault();
        e.stopPropagation();
        return false;
      }
      // Ctrl+U / Cmd+Opt+U (View Page Source)
      if (e.key === 'U' || e.key === 'u' || e.keyCode === 85) {
        e.preventDefault();
        e.stopPropagation();
        return false;
      }
      // Ctrl+S / Cmd+S (Save Page Source)
      if (e.key === 'S' || e.key === 's' || e.keyCode === 83) {
        e.preventDefault();
        e.stopPropagation();
        return false;
      }
    }
  }, { capture: true });

  // 3. Corporate Security & Legal Warning in Developer Console
  try {
    const warningStyle = 'font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; font-weight: bold; color: #d1002d;';
    const bodyStyle = 'font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; color: #1d1d1f;';
    
    console.log('%c[SECURITY WARNING] HRL International Private Limited™', warningStyle);
    console.log('%cThis web application and underlying frontend architecture are the proprietary intellectual property of Pavan Kumar Sadashiv.', bodyStyle);
    console.log('%cUnauthorized inspection, scraping, reverse-engineering, or asset copying is strictly prohibited under the Indian Copyright Act, 1957 and international IP laws.', bodyStyle);
  } catch (err) {
    // Fail silently in restricted sandbox
  }

})();
