param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

Add-Type -AssemblyName System.Drawing

$figureRoot = Join-Path $ProjectRoot 'figures\original'
$outputPath = Join-Path $ProjectRoot 'figures\square_overlap_compact.png'
$gap = 32

$paths = @(
    (Join-Path $figureRoot 'SH_2.png'),
    (Join-Path $figureRoot 'SH_2_1.png'),
    (Join-Path $figureRoot 'HP_2.png'),
    (Join-Path $figureRoot 'HP_2_1.png')
)

foreach ($path in $paths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing source figure: $path"
    }
}

$images = @($paths | ForEach-Object { [System.Drawing.Image]::FromFile($_) })
try {
    # The original panels place velocity on the left and pressure on the right.
    # Keep only the left half and the rows needed in the main paper:
    # reference, E2-selected Hopf specialist, and proposed T2-C. Full panels
    # remain untouched in figures/original and are used by the appendix.
    # Every retained band is drawn at native resolution, without interpolation.
    $cropWidths = @($images | ForEach-Object { [int][Math]::Floor($_.Width / 2) })
    [int]$columnWidth = ($cropWidths | Measure-Object -Maximum).Maximum
    $fieldBands = @(
        @(0, 657),       # title + CFD reference
        @(1232, 564),    # Hopf specialist
        @(1802, 563),    # proposed T2-C
        @(2365, 381)     # velocity colorbar
    )
    $errorBands = @(
        @(0, 93),        # title
        @(674, 574),     # Hopf-specialist error
        @(1256, 574),    # proposed T2-C error
        @(1830, 410)     # error colorbar
    )
    $bandGap = 6
    [int]$topHeight = ($fieldBands | ForEach-Object { $_[1] } | Measure-Object -Sum).Sum + $bandGap * ($fieldBands.Count - 1)
    [int]$bottomHeight = ($errorBands | ForEach-Object { $_[1] } | Measure-Object -Sum).Sum + $bandGap * ($errorBands.Count - 1)
    [int]$canvasWidth = 2 * $columnWidth + $gap
    [int]$canvasHeight = $topHeight + $gap + $bottomHeight

    $bitmap = [System.Drawing.Bitmap]::new($canvasWidth, $canvasHeight)
    $bitmap.SetResolution(400, 400)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.Clear([System.Drawing.Color]::White)

        $drawBands = {
            param($image, [int]$sourceWidth, [int]$destinationX, [int]$destinationY, $bands)
            [int]$currentY = $destinationY
            foreach ($band in $bands) {
                [int]$sourceY = $band[0]
                [int]$height = $band[1]
                $source = New-Object System.Drawing.Rectangle(0, $sourceY, $sourceWidth, $height)
                $destination = New-Object System.Drawing.Rectangle($destinationX, $currentY, $sourceWidth, $height)
                $graphics.DrawImage($image, $destination, $source, [System.Drawing.GraphicsUnit]::Pixel)
                $currentY += $height + $bandGap
            }
        }

        & $drawBands $images[0] $cropWidths[0] 0 0 $fieldBands
        & $drawBands $images[1] $cropWidths[1] 0 ($topHeight + $gap) $errorBands
        & $drawBands $images[2] $cropWidths[2] ($columnWidth + $gap) 0 $fieldBands
        & $drawBands $images[3] $cropWidths[3] ($columnWidth + $gap) ($topHeight + $gap) $errorBands

        # The source title band intersects a few pixels of its vertical row
        # label at the far left. Clear those detached glyph fragments while
        # leaving the retained row labels and scientific panels untouched.
        $errorTitleY = $topHeight + $gap
        $graphics.FillRectangle([System.Drawing.Brushes]::White, 0, $errorTitleY, 110, $errorBands[0][1])
        $graphics.FillRectangle([System.Drawing.Brushes]::White, ($columnWidth + $gap), $errorTitleY, 110, $errorBands[0][1])
    }
    finally {
        $graphics.Dispose()
    }

    $bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $bitmap.Dispose()
}
finally {
    foreach ($image in $images) {
        $image.Dispose()
    }
}

Write-Output $outputPath
