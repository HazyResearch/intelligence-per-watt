# Intelligence Per Watt Verified Submission Guide


## Submission Workflows

### Unverified Submissions
- **Website Form:** Submit via [Intelligence Per Watt website](https://trafficbenchv1.com/leaderboard/submit)
- **CLI Tool:** (Coming soon - TBD)

### Verified Submissions
- **GitHub PR:** Submit a Pull Request to REPOX with your benchmark results
- **Optional Linking:** If you already have an unverified submission, include its ID in your JSON to link them together

## Submission Process

### 1. Run Your Benchmark

Complete your Intelligence Per Watt evaluation using your model and hardware configuration. Ensure you have:
- Accuracy scores for each economic category tested
- Hardware metrics (energy, latency, TTFT, power, etc.)

### 2. Prepare Your Submission File

Create a JSON file with your results. See `example_org/example_submission.json` for the complete format and all available fields.

**Field Requirements:**

| Field | Required | Description |
|-------|----------|-------------|
| `organization` | **Yes** | Your org name (max 15 characters for display) |
| `contact_email` | **Yes** | Contact email for questions |
| `model_name` | **Yes** | Model name (e.g., "Qwen3-8B", "YourModel-7B") |
| `hardware_type` | **Yes** | Hardware used (e.g., "A100", "H200", "M4 Max") |
| `task_type` | **Yes** | One of: "reasoning", "chat", or "both" |
| `categoryMetrics` | **Yes** | Must contain at least one category with metrics |
| `unverified_submission_id` | No | Only if linking to existing website form submission |
| `paper_link` | No | Link to your paper/report |
| `code_link` | No | Link to your implementation code |
| `notes` | No | Implementation details (max 500 chars) |

**Category Metrics:**

Each category you tested should include:
- `accuracy` (required): Percentage (0-100)
- `avg_energy`: Average energy consumption in Joules
- `avg_compute`: Average compute in TFLOPS
- `avg_latency`: Average latency in milliseconds
- `avg_ttft`: Average time to first token in milliseconds
- `avg_power`: Average power consumption in Watts
- `avg_memory_bandwidth`: Average memory bandwidth in GB/s
- `avg_memory_bandwidth_peak`: Peak memory bandwidth in GB/s
- `avg_input_tokens`: Average input tokens per query
- `avg_output_tokens`: Average output tokens per response

**Available Economic Categories:**
- Computer and mathematical
- Arts, design, sports, entertainment, and media
- Life, physical, and social science
- Education instruction and library
- Architecture and engineering
- Business and financial operations
- Healthcare practitioners and technical
- Office and administrative support
- Legal services
- Community and social service
- Transportation and material moving
- Sales and related
- Food preparation and serving related
- General management
- Installation, maintenance, and repair
- Farming, fishing, and forestry
- Protective service
- Construction and extraction
- Healthcare support
- Production services
- Personal care and service
- Building grounds cleaning and maintenance

### 3. Fork and Clone REPOX

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/REPOX.git
cd REPOX
```

### 4. Create Your Submission Folder

Follow this folder structure:

```
REPOX/
└── submissions/
    └── your-organization-name/
        └── model-name_hardware-type_YYYYMMDD.json
```

**Example:**
```
submissions/uq/qwen3-8b_a100_20241104.json
```

**Naming Convention:**
- Folder: Lowercase organization name (use hyphens for spaces)
- File: `{model}_{hardware}_{date}.json`
- Use lowercase and hyphens
- Date format: YYYYMMDD

```bash
# Create your organization folder
mkdir -p submissions/your-org-name

# Add your submission file
cp your-results.json submissions/your-org-name/model-name_hardware-type_20241104.json

# Commit your changes
git add submissions/your-org-name/
git commit -m "Add verified results: ModelName on Hardware by YourOrg"
```

### 5. Submit Pull Request

Push to your fork and create a PR:

```bash
git push origin main
```

Then on GitHub:
1. Navigate to your forked repository
2. Click "Contribute" → "Open Pull Request"
3. Title: `Add verified results: [Model] on [Hardware] by [Organization]`
4. Description should include:
   - Brief description of your implementation
   - Link to your code repository
   - Link to paper (if available)
   - Any special notes about your setup

### 6. Review Process

Maintainers will:
1. Verify your submission format
2. Check that code is publicly accessible
3. Validate metric values are reasonable
4. Run spot-checks if needed
5. Merge your PR
6. Mark your submission as verified in the leaderboard

**Timeline:** Most PRs are reviewed within 3-5 business days.

## Quick Reference

**Example File:**
`example_org/example_submission.json` shows a complete example with all fields.

**Optional Field:**
- `unverified_submission_id`: Use this if you previously submitted via the website form and want to link this verified submission to it. If you don't have one, you can omit this field or set it to `null`.

**When to use `unverified_submission_id`:**
- ✅ You already submitted via the website form and want to update it with verified data
- ✅ You want to link your unverified and verified submissions together
- ❌ This is your first submission (omit this field or set to `null`)
- ❌ You're not linking to a previous website form submission (omit this field or set to `null`)

**Folder structure:**
```
submissions/
├── stanford/
│   ├── llama2-7b_a100_20241015.json
│   └── llama2-13b_h200_20241020.json
├── uq/
│   └── qwen3-8b_a100_20241104.json
└── mit/
    └── custom-model_mi300x_20241101.json
```

## FAQs

**Q: Can I submit multiple model/hardware combinations?**  
A: Yes! Create separate files for each combination in your organization folder.

**Q: What if I don't have all metrics?**  
A: Accuracy is required for each category. Other metrics are optional but strongly encouraged.

**Q: Can I update a submission?**  
A: Yes, submit a new PR updating your file. Include reasoning in the PR description.

**Q: What's the difference between verified and unverified?**  
A: Verified submissions require code and go through PR review. We verify the results so they can be trusted. Unverified submissions use the quick submission form on the website to see where you'd place.

**Q: Do I need to submit via both the website form AND GitHub PR?**  
A: No, choose one:
- **GitHub PR (verified):** For official verified results with code review
- **Website form (unverified):** For quick sharing without code review

**Q: Can I link my website submission with my GitHub PR?**  
A: Yes! If you previously submitted via the website form, you can link your GitHub PR submission to it:
1. Submit via the website form first and save your submission ID (e.g., `1730764800000_abc123xyz`)
2. In your JSON file, include `unverified_submission_id: "1730764800000_abc123xyz"`
3. When maintainers verify your PR, your original submission will be automatically updated and marked as verified

This is optional - you can submit directly via GitHub PR without a website submission.

## Support

Questions? Contact us:
- **Email:** ipw@scalingintelligence.org
- **GitHub Issues:** Open an issue in REPOX
- **Website:** https://trafficbenchv1.com/contact
