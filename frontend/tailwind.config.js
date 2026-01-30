/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        belief: {
          active: '#10B981',
          decaying: '#F59E0B',
          deprecated: '#6B7280',
          mutated: '#8B5CF6',
        },
        tension: {
          low: '#10B981',
          medium: '#F59E0B',
          high: '#EF4444',
        },
      },
    },
  },
  plugins: [],
};
