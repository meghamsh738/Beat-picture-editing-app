import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 45_000,
  use: {
    baseURL: 'http://localhost:4178',
    trace: 'on-first-retry',
    headless: true,
    viewport: { width: 1400, height: 900 },
    video: 'on'
  },
  webServer: {
    command: 'npm run dev -- --host --port 4178',
    url: 'http://localhost:4178',
    reuseExistingServer: true,
    stdout: 'ignore',
    stderr: 'pipe',
    timeout: 30_000
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    }
  ]
})
