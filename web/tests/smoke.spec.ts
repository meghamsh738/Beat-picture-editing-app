import { test, expect, type Page } from '@playwright/test'
import path from 'path'

const timelineTrackLane = '.track-lane'

const sampleAudioPath = path.join(process.cwd(), 'public', 'samples', 'sample-3s.wav')

const waitForRenderIdle = async (page: Page) => {
  await page.waitForTimeout(400)
}

test('timeline basics, snapping, loop handles, asset drop, export', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Premiere-style timeline prototype')).toBeVisible()

  // marker click moves playhead
  await page.getByText('Markers').scrollIntoViewIfNeeded()
  const firstMarker = page.locator('.marker-list li').first()
  await firstMarker.click()
  const playhead = page.locator('.playhead')
  await expect(playhead).toBeVisible()

  // drag a clip to trigger snap ghost
  const clip = page.locator('.clip').first()
  const box = await clip.boundingBox()
  if (box) {
    await page.mouse.move(box.x + 10, box.y + box.height / 2)
    await page.mouse.down()
    await page.mouse.move(box.x + 40, box.y + box.height / 2, { steps: 4 })
    await expect(page.locator('.snap-ghost')).toBeVisible()
    await page.mouse.up()
  }

  // loop handles respond
  const startHandle = page.locator('.loop-handle.start')
  await expect(startHandle).toBeVisible()
  await startHandle.click({ force: true })

  // asset upload + drag to track
  await page.getByRole('button', { name: 'Assets' }).click()
  const input = page.locator('input[type="file"]')
  await input.setInputFiles(sampleAudioPath)
  await waitForRenderIdle(page)
  const assetRow = page.locator('.asset-row').first()
  await expect(assetRow).toBeVisible()
  const sendButton = assetRow.getByRole('button', { name: /Send to/ }).first()
  await sendButton.click()
  await page.getByRole('button', { name: 'Edit' }).click()
  await waitForRenderIdle(page)
  const clipCount = await page.locator('.clip').count()
  expect(clipCount).toBeGreaterThan(5)

  // export preset mock
  await page.getByText('Export').click()
  await page.locator('select.ghost').selectOption('mp4')
  const renderBtn = page.getByRole('button', { name: 'Render preset' })
  await renderBtn.click()
  await page.waitForTimeout(600)

  await page.screenshot({ path: 'screenshots/timeline.png', fullPage: true })
})
